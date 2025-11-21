# src/models/trainer_custom/hybrid_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from configs.model.model_configs import MyModelConfig
from src.models.core.HDBE import HybridDoubleBranchEncoder
from src.utils.loss import info_nce_loss
from src.utils.metrics import compute_multitask_classification_metrics


class HybridModelTrainer(BaseTrainer):
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

        self.device = model_config.device
        self.lambda_recon = model_config.lambda_recon
        self.lambda_contrast = model_config.lambda_contrast
        self.contrastive_temperature = model_config.contrastive_temperature
        self.training_mode = model_config.training_mode    # "pretrain" or "finetune"

        self.model: HybridDoubleBranchEncoder | None = None
        self._loaded_model_path: Path | None = None

    ########################################################################
    # Batch 준비
    ########################################################################

    def _prepare_batch(self, batch):
        x = batch["x"].to(self.device)
        x_clean = batch["x_clean"].to(self.device)
        y = batch["y"].to(self.device)

        # BEMV 생성
        nan_mask = torch.isnan(x)
        bemv = (~nan_mask).to(x.dtype)

        return x, x_clean, y, bemv

    ########################################################################
    # Pretrain Epoch
    ########################################################################

    def _run_epoch_pretrain(
        self,
        data_loader,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None,
        phase: str
    ):
        if is_train:
            self.model.train()
            desc = f"Train ({phase})"
        else:
            self.model.eval()
            desc = f"Valid ({phase})"

        total_loss = 0.0
        num_batches = 0

        # tqdm 적용
        for batch in tqdm(data_loader, desc=desc):
            x, x_clean, y, bemv = self._prepare_batch(batch)

            bemv_clean = torch.ones_like(bemv)

            with torch.set_grad_enabled(is_train):
                out_masked = self.model(x, bemv)
                out_clean = self.model(x_clean, bemv_clean)

                z_masked = out_masked["latent"]
                z_clean = out_clean["latent"]

                info_loss = info_nce_loss(
                    z_clean,
                    z_masked,
                    self.contrastive_temperature
                )

                recon_loss = F.mse_loss(out_masked["recon"], x_clean)

                loss = (
                    self.lambda_contrast * info_loss
                    + self.lambda_recon * recon_loss
                )

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    ########################################################################
    # Finetune Epoch
    ########################################################################

    def _run_epoch_finetune(
        self,
        data_loader,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None,
        phase: str,
        return_outputs: bool = False,
    ):
        criterion = nn.CrossEntropyLoss()

        if is_train:
            self.model.train()
            desc = f"Train ({phase})"
        else:
            self.model.eval()
            desc = f"Valid ({phase})"

        total_loss = 0.0
        num_batches = 0

        if return_outputs:
            all_preds = []
            all_labels = []

        for batch in tqdm(data_loader, desc=desc):
            x, x_clean, y, bemv = self._prepare_batch(batch)
            x = torch.nan_to_num(x, nan=0.0)

            with torch.set_grad_enabled(is_train):
                out = self.model(x, bemv)

                preds = out["preds"]   # (B, H, C)
                B, H, C = preds.shape

                preds_flat = preds.reshape(B * H, C)
                y_flat = y.reshape(B * H).long()

                loss = criterion(preds_flat, y_flat)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if return_outputs:
                pred_labels = preds.argmax(dim=-1).detach().cpu()  # (B, H)
                all_preds.append(pred_labels)
                all_labels.append(y.detach().cpu())

        avg_loss = total_loss / max(1, num_batches)

        if return_outputs:
            preds_all = torch.cat(all_preds, dim=0).numpy()    # (N, H)
            labels_all = torch.cat(all_labels, dim=0).numpy()  # (N, H)
            return avg_loss, preds_all, labels_all

        return avg_loss



    ########################################################################
    # 공용 _run_epoch : training_mode에 따라 분기
    ########################################################################

    def _run_epoch(
        self,
        *,
        data_loader,
        is_train,
        optimizer,
        phase,
    ):
        if self.training_mode == "pretrain":
            return self._run_epoch_pretrain(
                data_loader=data_loader,
                is_train=is_train,
                optimizer=optimizer,
                phase=phase,
            )
        else:
            return self._run_epoch_finetune(
                data_loader=data_loader,
                is_train=is_train,
                optimizer=optimizer,
                phase=phase,
            )

    ########################################################################
    # fit()
    ########################################################################

    def fit(self, train_dataset, valid_dataset):
        tr_loader = self.make_loader(train_dataset, shuffle=True)
        vl_loader = self.make_loader(valid_dataset, shuffle=False)

        self.model = HybridDoubleBranchEncoder(
            input_dim=train_dataset.feature_dim,
            embed_dim=self.model_config.n_d,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.model_config.max_epochs - 1,
            eta_min=self.model_config.min_learning_rate
        )

        best_val_loss = None
        best_state = None
        patience_counter = 0
        patience = self.model_config.patience_count

        history = {
            "train_loss": [],
            "valid_loss": [],
            "lr": [],
        }

        for epoch in range(self.model_config.max_epochs):
            train_loss = self._run_epoch(
                data_loader=tr_loader,
                is_train=True,
                optimizer=optimizer,
                phase="train"
            )

            val_loss = self._run_epoch(
                data_loader=vl_loader,
                is_train=False,
                optimizer=None,
                phase="valid"
            )

            print(f"[Epoch {epoch+1}] train={train_loss:.5f}, valid={val_loss:.5f}")

            history["train_loss"].append(float(train_loss))
            history["valid_loss"].append(float(val_loss))
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            scheduler.step()

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    ########################################################################
    # eval()
    ########################################################################

    def eval(self, test_dataset):
        te_loader = self.make_loader(test_dataset, shuffle=False)

        if self.model is None:
            if self._loaded_model_path is not None:
                self.model = HybridDoubleBranchEncoder(
                    input_dim=test_dataset.feature_dim,
                    embed_dim=self.model_config.n_d,
                ).to(self.device)
                self.model.load_state_dict(
                    torch.load(self._loaded_model_path, map_location=self.device)
                )
                self.model.eval()
            else:
                raise RuntimeError("모델이 로드되지 않았습니다. fit() 또는 load() 먼저 실행해야 합니다.")

        # pretrain 모드: reconstruction loss만
        if self.training_mode == "pretrain":
            loss = self._run_epoch(
                data_loader=te_loader,
                is_train=False,
                optimizer=None,
                phase="valid"
            )
            return {"loss": float(loss)}

        # finetune 모드: _run_epoch_finetune에서 루프 + preds/labels까지 처리
        loss, preds_all, labels_all = self._run_epoch_finetune(
            data_loader=te_loader,
            is_train=False,
            optimizer=None,
            phase="test",
            return_outputs=True,
        )

        results = compute_multitask_classification_metrics(preds_all, labels_all)
        results["loss"] = float(loss)

        return results

    ########################################################################
    # Save / Load
    ########################################################################

    def save(self, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
        out_path = save_path / self.model_config.save_model_name
        torch.save(self.model.state_dict(), out_path)
        return out_path

    def load(self, model_path: Path):
        self._loaded_model_path = model_path
        return model_path
