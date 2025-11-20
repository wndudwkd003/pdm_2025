# src/models/trainer/custom_trainer/my_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from tqdm.auto import tqdm
from pathlib import Path
from typing import Any

from configs.model.model_configs import MyModelConfig
from src.models.core.my_model import MyModel
from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.utils.metrics import compute_multitask_classification_metrics
from src.utils.loss import info_nce_loss


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
        self.lambda_contrast = model_config.lambda_contrast
        self.contrastive_temperature = model_config.contrastive_temperature
        self.training_mode = model_config.training_mode
        self.device = model_config.device
        self.model: MyModel | None = None
        self.base_model: MyModel | None = None
        self._loaded_model_path: Path | None = None

    def _prepare_batch(self, batch):
        x = batch["x"].to(self.device)
        x_clean = batch["x_clean"].to(self.device)
        y = batch["y"].to(self.device)

        nan_mask = torch.isnan(x)
        bemv = (~nan_mask).to(x.dtype)

        x_img = batch["x_img"]
        if x_img is not None:
            x_img = x_img.to(self.device)

        return x, x_clean, x_img, y, bemv

    def _run_epoch_pretrain(
        self,
        model: nn.Module,
        data_loader,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None,
        phase: str,
        epoch_idx: int,
        max_epoch: int,
    ):
        if phase == "train":
            model.train()
            desc = f"Train {epoch_idx}/{max_epoch}"
        else:
            model.eval()
            desc = f"Valid {epoch_idx}/{max_epoch}"

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(data_loader, desc=desc):
            x, x_clean, x_img, y, bemv = self._prepare_batch(batch)
            bemv_clean = torch.ones_like(bemv, dtype=bemv.dtype)

            with torch.set_grad_enabled(is_train):
                out_masked = model(x, bemv, x_img)
                out_clean = model(x_clean, bemv_clean, x_img)

                z_masked = out_masked["latent"]
                z_clean = out_clean["latent"]

                info_loss = info_nce_loss(
                    z_clean,
                    z_masked,
                    self.contrastive_temperature
                )

                M_loss_total = 0.5 * (
                    out_masked["M_loss_total"] + out_clean["M_loss_total"]
                )
                reg_loss = - self.lambda_sparse * M_loss_total

                loss = self.lambda_contrast * info_loss + reg_loss

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _run_epoch_finetune(
        self,
        model: nn.Module,
        data_loader,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None,
        phase: str,
        epoch_idx: int,
        max_epoch: int,
    ):
        num_tasks = len(self.model_config.output_dims)
        criterion = nn.CrossEntropyLoss()

        if phase == "train":
            model.train()
            desc = f"Train {epoch_idx}/{max_epoch}"
        else:
            model.eval()
            desc = f"Valid {epoch_idx}/{max_epoch}"

        total_loss = 0.0
        num_batches = 0
        all_targets = []
        all_preds = []

        for batch in tqdm(data_loader, desc=desc):
            x, x_clean, x_img, y, bemv = self._prepare_batch(batch)

            with torch.set_grad_enabled(is_train):
                if x_img is not None:
                    out = model(x, bemv, x_img=x_img)
                else:
                    out = model(x, bemv)

                outs = out["outs"]
                M_loss_total = out["M_loss_total"]

                sup_loss = 0.0

                for task_idx, logits in enumerate(outs):
                    y_task = y[:, task_idx].long()
                    sup_loss += criterion(logits, y_task)

                sup_loss = sup_loss / num_tasks
                reg_loss = - self.lambda_sparse * M_loss_total
                loss = sup_loss + reg_loss

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

            if not is_train:
                pred_list = []
                for logits in outs:
                    pred_list.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_preds.append(np.stack(pred_list, axis=1))
                all_targets.append(y.cpu().numpy())

        if is_train:
            return total_loss / max(1, num_batches)

        if len(all_preds) == 0:
            return {"num_samples": 0}

        y_pred = np.concatenate(all_preds, axis=0).astype(np.int64)
        y_true = np.concatenate(all_targets, axis=0).astype(np.int64)

        metrics = compute_multitask_classification_metrics(y_true, y_pred)
        metrics["num_samples"] = y_true.shape[0]
        metrics["loss"] = total_loss / max(1, num_batches)
        return metrics

    def _run_epoch(
        self,
        *,
        model: nn.Module,
        data_loader,
        is_train: bool,
        optimizer: torch.optim.Optimizer | None,
        phase: str,
        epoch_idx: int,
        max_epoch: int,
    ):
        if self.training_mode == "pretrain":
            return self._run_epoch_pretrain(
                model=model,
                data_loader=data_loader,
                is_train=is_train,
                optimizer=optimizer,
                phase=phase,
                epoch_idx=epoch_idx,
                max_epoch=max_epoch,
            )
        return self._run_epoch_finetune(
            model=model,
            data_loader=data_loader,
            is_train=is_train,
            optimizer=optimizer,
            phase=phase,
            epoch_idx=epoch_idx,
            max_epoch=max_epoch,
        )

    def fit(self, train_dataset, valid_dataset):
        history = {
            "train_loss": [],
            "valid_loss": [],
            "val_metrics": [],
            "num_samples": [],
            "lr": [],
        }

        best_val_loss = None
        epochs_no_improve = 0
        patience = self.model_config.patience_count
        best_state_dict = None

        tr_loader = self.make_loader(train_dataset, shuffle=True)
        vl_loader = self.make_loader(valid_dataset, shuffle=False)

        self.base_model = self._model(
            model_config=self.model_config,
            feature_dim=train_dataset.feature_dim,
        )
        self.model = self.base_model
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

        max_epoch = self.model_config.max_epochs

        for epoch in range(max_epoch):
            epoch_idx = epoch + 1

            train_loss = self._run_epoch(
                model=model,
                data_loader=tr_loader,
                is_train=True,
                optimizer=optimizer,
                phase="train",
                epoch_idx=epoch_idx,
                max_epoch=max_epoch,
            )

            if self.training_mode == "pretrain":
                val_loss = self._run_epoch(
                    model=model,
                    data_loader=vl_loader,
                    is_train=False,
                    optimizer=None,
                    phase="valid",
                    epoch_idx=epoch_idx,
                    max_epoch=max_epoch,
                )
                valid_loss = float(val_loss)
                val_metrics = {}
                num_samples = 0
            else:
                val_metrics = self._run_epoch(
                    model=model,
                    data_loader=vl_loader,
                    is_train=False,
                    optimizer=None,
                    phase="valid",
                    epoch_idx=epoch_idx,
                    max_epoch=max_epoch,
                )
                valid_loss = float(val_metrics["loss"])
                num_samples = int(val_metrics["num_samples"])

            print(
                f"[Epoch {epoch_idx}] "
                f"Train loss: {train_loss:.6f}  "
                f"Valid loss: {valid_loss:.6f}"
            )

            current_lr = optimizer.param_groups[0]['lr']
            history["lr"].append(float(current_lr))
            history["train_loss"].append(float(train_loss))
            history["valid_loss"].append(float(valid_loss))

            metrics_float = {}
            for k in val_metrics:
                v = val_metrics[k]
                metrics_float[k] = float(v)
            history["val_metrics"].append(metrics_float)
            history["num_samples"].append(num_samples)

            if best_val_loss is None or valid_loss < best_val_loss:
                best_val_loss = valid_loss
                epochs_no_improve = 0
                best_state_dict = {}
                state_dict = self.base_model.state_dict()
                for key in state_dict:
                    value = state_dict[key]
                    best_state_dict[key] = value.detach().cpu().clone()
            else:
                epochs_no_improve = epochs_no_improve + 1

            scheduler.step()

            if best_val_loss is not None and epochs_no_improve >= patience:
                print(
                    f"[EarlyStopping] patience={patience}에 도달하여 "
                    f"epoch {epoch_idx}에서 학습을 종료합니다."
                )
                break

        if best_state_dict is not None:
            self.base_model.load_state_dict(best_state_dict)

        return history

    def _model(self, model_config: MyModelConfig, feature_dim: int):
        model = MyModel(
            input_dim=feature_dim,
            output_dim=model_config.output_dims,
            n_steps=model_config.n_steps,
            multimodal_setting=model_config.multimodal_setting,
            nhead=model_config.nhead,
            ff_dim=model_config.ff_dim,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout,
            max_seq_len=model_config.max_seq_len,
        ).to(self.device)
        return model

    def eval(self, test_dataset) -> dict[str, Any]:
        te_loader = self.make_loader(test_dataset, shuffle=False)

        if self.model is None:
            if self._loaded_model_path is None:
                raise RuntimeError("load()를 먼저 호출하거나 fit()을 먼저 실행해야 합니다.")

            self.base_model = self._model(
                model_config=self.model_config,
                feature_dim=test_dataset.feature_dim,
            )

            weight_file = self._loaded_model_path / self.model_config.save_model_name
            state_dict = torch.load(weight_file, map_location=self.device)
            self.base_model.load_state_dict(state_dict)
            self.model = self.base_model

        if self.training_mode == "pretrain":
            loss = self._run_epoch(
                model=self.model,
                data_loader=te_loader,
                is_train=False,
                optimizer=None,
                phase="valid",
                epoch_idx=1,
                max_epoch=1,
            )
            return {"loss": float(loss)}

        metrics = self._run_epoch(
            model=self.model,
            data_loader=te_loader,
            is_train=False,
            optimizer=None,
            phase="valid",
            epoch_idx=1,
            max_epoch=1,
        )
        return metrics

    def save(self, save_path: Path):
        if self.base_model is None:
            raise RuntimeError("Model is not trained yet.")

        save_path.mkdir(parents=True, exist_ok=True)
        out = save_path / self.model_config.save_model_name
        torch.save(self.base_model.state_dict(), out)
        return out

    def load(self, model_path: Path):
        self._loaded_model_path = model_path
        return model_path
