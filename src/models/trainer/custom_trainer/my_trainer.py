# src/models/trainer_custom/mptms/tabnet_trainer.py

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from src.utils.metrics import compute_multitask_classification_metrics
from configs.model.model_configs import MyModelConfig

from src.models.core.my_model import MyModel
from src.models.core.module.attention_viz import log_attention

from torch.optim.lr_scheduler import CosineAnnealingLR

class MyModelTrainer(BaseTrainer):
    model_config: MyModelConfig

    def __init__(
        self,
        *,
        work_dir: Path,
        data_collator: BaseCollator,
        model_config: MyModelConfig,
        metadata: dict[str, Any] = None
    ):
        super().__init__(
            work_dir=work_dir,
            data_collator=data_collator,
            model_config=model_config,
            metadata=metadata
        )

        self.lambda_sparse = model_config.lambda_sparse
        self.lambda_recon = model_config.lambda_recon

        self.device = model_config.device
        self.model: MyModel | None = None
        self._loaded_model_path: Path | None = None

        self._attn_cache: dict[str, dict[str, Any] | None] = {
            "train": None,
            "valid": None,
        }

    # ---------------------------------------------------------
    # Batch 공통 처리
    # ---------------------------------------------------------
    def _prepare_batch(self, batch):
        x = batch["x"].to(self.device)               # 마스킹된 입력
        x_original = batch["x_original"].to(self.device)  # 마스킹 전 원본
        y = batch["y"].to(self.device)

        nan_mask = torch.isnan(x)
        bemv = (~nan_mask).to(x.dtype)

        x_img = batch.get("x_img", None)
        if x_img is not None:
            x_img = x_img.to(self.device)

        return x, x_original, x_img, y, bemv


    # ---------------------------------------------------------
    # train / valid epoch
    # ---------------------------------------------------------
    def _run_epoch(
        self,
        *,
        model: nn.Module,
        data_loader,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None = None,
        need_log: bool = False,
        phase: str = "train",
    ):
        num_tasks = len(self.model_config.output_dims)
        criterion = nn.CrossEntropyLoss()

        if phase == "train":
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        num_batches = 0
        all_targets = []
        all_preds = []

        for batch in data_loader:
            x, x_original, x_img, y, bemv = self._prepare_batch(batch)

            with torch.set_grad_enabled(is_train):
                # ------------------------------
                # 모델 forward
                # ------------------------------
                if x_img is not None:
                    outs, M_loss, tr_attn_maps = model(x, bemv, x_img=x_img)
                else:
                    outs, M_loss, tr_attn_maps = model(x, bemv)

                # ------------------------------
                # Attention 캐시 (logging용)
                # ------------------------------
                if need_log:
                    # MPD/MPIE encoder에서 나오는 attention
                    mpie_enc_maps = tr_attn_maps.get("mpie_attention_maps", None)
                    # 여기서는 MPIE용 Transformer attention을 tr_attention_maps로 사용
                    tr_maps = tr_attn_maps.get("mpie", None)

                    mpie_list_cpu = None
                    if mpie_enc_maps is not None:
                        mpie_list_cpu = [A.detach().cpu() for A in mpie_enc_maps]

                    tr_list_cpu = None
                    if tr_maps is not None:
                        tr_list_cpu = [A.detach().cpu() for A in tr_maps]

                    self._attn_cache[phase] = {
                        "x": x.detach().cpu(),
                        "bemv": bemv.detach().cpu(),
                        "mpie_attention_maps": mpie_list_cpu,
                        "tr_attention_maps": tr_list_cpu,
                    }
                    need_log = False

                # ------------------------------
                # supervised loss
                # ------------------------------
                sup_loss = 0.0
                for task_idx, logits in enumerate(outs):
                    sup_loss += criterion(logits, y[:, task_idx].long())
                sup_loss /= num_tasks

                # ------------------------------
                # sparsity 정규화 (MPIE + MPDE 합산)
                # ------------------------------
                reg_loss = - self.lambda_sparse * M_loss


                # 최종 loss
                loss = sup_loss + reg_loss

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # ------------------------------
            # validation용 prediction 저장
            # ------------------------------
            if not is_train:
                with torch.no_grad():
                    pred_list = [
                        torch.argmax(logits, dim=1).cpu().numpy()
                        for logits in outs
                    ]
                    all_preds.append(np.stack(pred_list, axis=1))
                    all_targets.append(y.cpu().numpy())

        if is_train:
            return total_loss / max(1, num_batches)

        # ------------------------------
        # evaluation metrics
        # ------------------------------
        if len(all_preds) == 0:
            return {"num_samples": 0}

        y_pred = np.concatenate(all_preds, axis=0).astype(np.int64)
        y_true = np.concatenate(all_targets, axis=0).astype(np.int64)

        metrics = compute_multitask_classification_metrics(y_true, y_pred)
        metrics["num_samples"] = y_true.shape[0]
        metrics["loss"] = total_loss / max(1, num_batches)
        return metrics


    # ---------------------------------------------------------
    # 학습 루프
    # ---------------------------------------------------------
    def fit(self, train_dataset, valid_dataset):

        # ---------------------------
        # history dict 생성
        # ---------------------------
        history = {
            "train_loss": [],
            "valid_loss": [],
            "val_metrics": [],
            "num_samples": [],
            "lr": [],
        }


        tr_loader = self.make_loader(train_dataset, shuffle=True)
        vl_loader = self.make_loader(valid_dataset, shuffle=False)

        first_batch = next(iter(tr_loader))
        x0: torch.Tensor = first_batch["x"]
        _, _, F = x0.shape

        self.model = MyModel(
            input_dim=F,
            output_dim=self.model_config.output_dims,
            n_steps=self.model_config.n_steps,
            multimodal_setting=self.model_config.multimodal_setting,
        ).to(self.device)

        model = self.model

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.model_config.max_epochs - 1,
            eta_min=self.model_config.min_learning_rate,
        )

        attn_log_dir = self.work_dir / "attn_logs"
        attn_log_dir.mkdir(parents=True, exist_ok=True)
        attn_log_interval = 5

        for epoch in range(self.model_config.max_epochs):
            epoch_idx = epoch + 1
            need_log = (epoch_idx % attn_log_interval == 0)

            # train step
            train_loss = self._run_epoch(
                model=model,
                data_loader=tr_loader,
                is_train=True,
                optimizer=optimizer,
                need_log=need_log,
                phase="train",
            )
            print(f"[Epoch {epoch_idx}/{self.model_config.max_epochs}] "
                  f"train_loss={train_loss:.6f}")

            # valid step
            val_metrics = self._run_epoch(
                model=model,
                data_loader=vl_loader,
                is_train=False,
                optimizer=None,
                need_log=need_log,
                phase="valid",
            )

            # logging
            valid_loss = val_metrics.get("loss", None)
            msg_parts = []
            if isinstance(valid_loss, (float, int)):
                msg_parts.append(f"loss={valid_loss:.6f}")

            for k, v in val_metrics.items():
                if k in ("loss", "num_samples"):
                    continue
                if isinstance(v, (float, int)):
                    msg_parts.append(f"{k}={v:.6f}")

            print("[Val] " + "  ".join(msg_parts))

            current_lr = optimizer.param_groups[0]['lr']
            history["lr"].append(float(current_lr))

            history["train_loss"].append(float(train_loss))
            history["valid_loss"].append(float(valid_loss))
            history["val_metrics"].append({k: float(v) for k, v in val_metrics.items() if isinstance(v, (float, int))})
            history["num_samples"].append(int(val_metrics["num_samples"]))

            scheduler.step()

            # attention log
            if need_log:
                for phase in ("train", "valid"):
                    self._log_attention(
                        epoch=epoch_idx,
                        phase=phase,
                        attn_log_dir=attn_log_dir,
                        attn_type="mpie_attention_maps",
                    )
                    self._log_attention(
                        epoch=epoch_idx,
                        phase=phase,
                        attn_log_dir=attn_log_dir,
                        attn_type="tr_attention_maps",
                    )

        return history

    # ---------------------------------------------------------
    # Attention Log 저장
    # ---------------------------------------------------------
    def _log_attention(
        self,
        epoch: int,
        phase: str,
        attn_log_dir: Path,
        attn_type: str,
    ):
        cache = self._attn_cache.get(phase)
        if cache is None:
            print(f"[Attention Log] No cached attention for phase={phase} at epoch={epoch}")
            return

        x_dbg: torch.Tensor = cache["x"]
        bemv_dbg: torch.Tensor = cache["bemv"]

        log_attention(
            epoch=epoch,
            phase=phase,
            attn_log_dir=attn_log_dir,
            attn_type=attn_type,
            x_dbg=x_dbg,
            bemv_dbg=bemv_dbg,
            attention_maps_dbg=cache[attn_type],
            time_series_serialization_fn=self.model.time_series_serialization,
            device=self.device,
        )

    # ---------------------------------------------------------
    # eval/test
    # ---------------------------------------------------------
    def eval(self, test_dataset) -> dict[str, Any]:
        te_loader = self.make_loader(test_dataset, shuffle=False)

        if self.model is None:
            if self._loaded_model_path is None:
                raise RuntimeError("load()를 먼저 호출하거나 fit()을 먼저 실행해야 합니다.")

            first_batch = next(iter(te_loader))
            x0: torch.Tensor = first_batch["x"].to(self.device)
            _, _, F = x0.shape

            self.model = MyModel(
                input_dim=F,
                output_dim=self.model_config.output_dims,
                n_steps=self.model_config.n_steps,
                multimodal_setting=self.model_config.multimodal_setting,
            ).to(self.device)

            weight_file = self._loaded_model_path / self.model_config.save_model_name
            state_dict = torch.load(weight_file, map_location=self.device)
            self.model.load_state_dict(state_dict)

        metrics = self._run_epoch(
            model=self.model,
            data_loader=te_loader,
            is_train=False,
            optimizer=None,
            need_log=False,
            phase="valid",
        )
        return metrics

    # ---------------------------------------------------------
    # save / load
    # ---------------------------------------------------------
    def save(self, save_path: Path):
        if self.model is None:
            raise RuntimeError("Model is not trained yet.")

        save_path.mkdir(parents=True, exist_ok=True)
        out = save_path / self.model_config.save_model_name
        torch.save(self.model.state_dict(), out)
        return out

    def load(self, model_path: Path):
        self._loaded_model_path = model_path
        return model_path
