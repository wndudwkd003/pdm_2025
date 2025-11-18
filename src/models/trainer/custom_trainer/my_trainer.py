# src/models/trainer_custom/mptms/tabnet_trainer.py

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from src.utils.metrics import compute_multitask_classification_metrics
from configs.model.model_configs import MyModelConfig

from src.models.core.my_model import MyModel

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

    # ---------------------------------------------------------
    # Batch 공통 처리
    # ---------------------------------------------------------
    def _prepare_batch(self, batch):
        x = batch["x"].to(self.device)                    # 마스킹된 입력
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

        history = {
            "train_loss": [],
            "valid_loss": [],
            "val_metrics": [],
            "num_samples": [],
            "lr": [],
        }

        # early stopping 상태
        best_val_loss = None
        epochs_no_improve = 0
        patience = self.model_config.patience_count
        best_state_dict = None

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

        current_device = torch.cuda.current_device()

        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[current_device],
                output_device=current_device,
                find_unused_parameters=True,
            )
        else:
            print("Distributed training is not initialized.")

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

        if is_distributed:
            rank = dist.get_rank()
            is_main_rank = (rank == 0)
        else:
            is_main_rank = True

        for epoch in range(self.model_config.max_epochs):
            epoch_idx = epoch + 1

            if is_distributed:
                sampler = getattr(tr_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch_idx)

            train_loss = self._run_epoch(
                model=model,
                data_loader=tr_loader,
                is_train=True,
                optimizer=optimizer,
                phase="train",
            )

            if is_main_rank:
                print(
                    f"[Epoch {epoch_idx}/{self.model_config.max_epochs}] "
                    f"train_loss={train_loss:.6f}"
                )

            val_metrics = self._run_epoch(
                model=model,
                data_loader=vl_loader,
                is_train=False,
                optimizer=None,
                phase="valid",
            )

            valid_loss = val_metrics.get("loss", None)

            if is_main_rank:
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
                history["val_metrics"].append({
                    k: float(v)
                    for k, v in val_metrics.items()
                    if isinstance(v, (float, int))
                })
                history["num_samples"].append(int(val_metrics["num_samples"]))

                # ----------------------------
                # early stopping 상태 + best 모델 갱신
                # ----------------------------
                if isinstance(valid_loss, (float, int)):
                    if (best_val_loss is None) or (valid_loss < best_val_loss):
                        best_val_loss = valid_loss
                        epochs_no_improve = 0

                        # DDP 래핑 여부에 따라 실제 nn.Module 선택
                        model_for_save = self.model
                        if isinstance(self.model, DDP):
                            model_for_save = self.model.module

                        # state_dict 전체 복사해서 CPU에 보관
                        best_state_dict = {}
                        state_dict = model_for_save.state_dict()
                        for key, value in state_dict.items():
                            best_state_dict[key] = value.detach().cpu().clone()
                    else:
                        epochs_no_improve += 1

            scheduler.step()

            # --------------------------
            # early stopping 체크
            # --------------------------
            stop_training = False

            if is_main_rank and isinstance(valid_loss, (float, int)):
                if (best_val_loss is not None) and (epochs_no_improve >= patience):
                    print(
                        f"[EarlyStopping] patience={patience}에 도달하여 "
                        f"epoch {epoch_idx}에서 학습을 종료합니다."
                    )
                    stop_training = True

            if is_distributed:
                stop_tensor = torch.tensor(
                    [1 if stop_training else 0],
                    device=self.device,
                )
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break

        # --------------------------
        # 학습 종료 후 best model 로 복원
        # --------------------------
        if best_state_dict is not None:
            model_for_load = self.model
            if isinstance(self.model, DDP):
                model_for_load = self.model.module
            model_for_load.load_state_dict(best_state_dict)

        return history

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

        model_for_save = self.model
        if isinstance(self.model, DDP):
            model_for_save = self.model.module

        torch.save(model_for_save.state_dict(), out)
        return out

    def load(self, model_path: Path):
        self._loaded_model_path = model_path
        return model_path
