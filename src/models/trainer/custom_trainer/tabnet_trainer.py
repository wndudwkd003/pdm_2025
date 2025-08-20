# src/models/trainer_custom/mptms/tabnet_trainer.py
from pathlib import Path
from typing import Any
import numpy as np
from torch.utils.data import DataLoader
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer


class TabNetTrainer(BaseTrainer):
    def __init__(
        self,
        work_dir: Path,
        data_collator: BaseCollator,
        *,
        batch_size: int = 1024,
        max_epochs: int = 200,
        patience: int = 20,
        num_workers: int = 4,
        collect_batch_size: int = 64,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        gamma: float = 1.5,
        mask_type: str = "sparsemax",
    ):
        super().__init__(work_dir=work_dir, data_collator=data_collator)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.num_workers = num_workers
        self.collect_batch_size = collect_batch_size
        self.n_d, self.n_a, self.n_steps = n_d, n_a, n_steps
        self.gamma, self.mask_type = gamma, mask_type
        self.model = None

    def _dataset_to_numpy(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            dataset,
            batch_size=self.collect_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.data_collator,
            drop_last=False,
        )
        Xs: list[np.ndarray] = []
        Ys: list[np.ndarray] = []
        for batch in loader:
            Xs.append(batch["x"].cpu().numpy())          # (B, D)
            Ys.append(batch["y"].cpu().numpy())          # (B, T_out)
        X = np.concatenate(Xs, axis=0)                   # (N, D)
        Y = np.concatenate(Ys, axis=0).astype(np.int64)  # (N, T_out)
        return X, Y

    def fit(self, train_dataset, valid_dataset):
        X_tr, y_tr = self._dataset_to_numpy(train_dataset)
        X_va, y_va = self._dataset_to_numpy(valid_dataset)

        n_tasks = y_tr.shape[1]
        output_dim = [4] * n_tasks

        self.model = TabNetMultiTaskClassifier(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, mask_type=self.mask_type,
            output_dim=output_dim,
            device_name="cuda",
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_va, y_va)],
            eval_name=["train", "valid"],
            eval_metric=["accuracy"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=128,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return dict(self.model.history.history)

    def eval(self, test_dataset) -> dict[str, Any]:
        X_te, y_te = self._dataset_to_numpy(test_dataset)  # (N,D), (N,T_out)

        pred_list = self.model.predict(X_te)               # list[np.ndarray], len=T_out
        y_pred = np.stack([np.asarray(p).reshape(-1) for p in pred_list], axis=1).astype(np.int64)

        T = y_te.shape[1]
        per_step: list[float] = []
        for j in range(T):
            per_step.append(float((y_te[:, j] == y_pred[:, j]).mean()))
        overall = float((y_te == y_pred).mean())

        return {
            "overall_accuracy": overall,
            "per_step_accuracy": per_step,
            "num_samples": int(y_te.shape[0]),
            "num_tasks": int(T),
        }

    def save(self, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
        out = save_path / "save_model"
        self.model.save_model(str(out))  # save_model.zip
        return out

    def load(self, model_path: Path):
        self.model = TabNetMultiTaskClassifier(device_name="cuda")
        self.model.load_model(str(model_path / "save_model.zip"))
        return self.model
