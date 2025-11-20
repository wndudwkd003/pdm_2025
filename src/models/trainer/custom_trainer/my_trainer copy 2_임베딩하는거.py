from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from configs.model.model_configs import MyModelConfig
from src.models.core.HDBE import HybridDoubleBranchEncoder

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

        self.device = model_config.device
        self.lambda_recon = model_config.lambda_recon
        self.lambda_contrast = 1.0
        self.model: HybridDoubleBranchEncoder = None
        self._loaded_model_path: Path = None

    def _prepare_batch(self, batch):
        x = batch["x"].to(self.device)
        x_clean = batch["x_clean"].to(self.device)
        y = batch["y"].to(self.device)
        bemv = (~torch.isnan(x)).to(x.dtype)
        return x, x_clean, y, bemv


    def _convert_zero_nan(self, x: torch.Tensor, x_bemv: torch.Tensor):
        missing_mask = (x_bemv == 0)
        x = x.clone()
        x[missing_mask] = 0.0
        return x

    def _run_epoch(self, *, data_loader, is_train, optimizer=None, phase="train", lam_noise=0.1, temperature=0.1):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            x, x_clean, y, bemv = self._prepare_batch(batch)

            x_zero = self._convert_zero_nan(x, bemv)

            bemv_clean = torch.ones_like(bemv, dtype=bemv.dtype)

            with torch.set_grad_enabled(is_train):
                # Missing View, Original View
                out_clean = self.model(x_zero, bemv)
                out_noisy = self.model(x_clean, bemv_clean)

                # Reconstruction loss
                num_recon = out_noisy["recon"]
                recon_loss = F.mse_loss(num_recon, x_clean)

                # Contrastive loss
                z_clean = out_clean["latent"]
                z_noisy = out_noisy["latent"]
                info_loss = info_nce_loss(z_clean, z_noisy, temperature)

                loss = self.lambda_contrast * info_loss + self.lambda_recon * recon_loss

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)


    def fit(self, train_dataset, valid_dataset):
        tr_loader = self.make_loader(train_dataset, shuffle=True)
        vl_loader = self.make_loader(valid_dataset, shuffle=False)

        self.model = HybridDoubleBranchEncoder(
            input_dim=train_dataset.feature_dim,
            embed_dim=64,
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

        best_val_loss = float("inf")
        patience = self.model_config.patience_count
        patience_counter = 0
        best_state = None

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

            print(f"[Epoch {epoch+1}] train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")

            history["train_loss"].append(float(train_loss))
            history["valid_loss"].append(float(val_loss))
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            scheduler.step()

            if val_loss < best_val_loss:
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




    def eval(self, test_dataset):
        te_loader = self.make_loader(test_dataset, shuffle=False)
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. fit() 또는 load() 먼저 실행해야 합니다.")
        loss = self._run_epoch(data_loader=te_loader, is_train=False)
        return {"loss": float(loss)}



    def save(self, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
        out_path = save_path / self.model_config.save_model_name
        torch.save(self.model.state_dict(), out_path)
        return out_path

    def load(self, model_path: Path):
        self._loaded_model_path = model_path
        return model_path
