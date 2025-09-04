# src/models/trainer_custom/mptms/tabnet_trainer.py
from pathlib import Path
from typing import Any
import numpy as np
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
import torch

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from src.utils.metrics import compute_multitask_classification_metrics
from configs.model.model_configs import MyModelConfig

from src.models.core.my_model import MyModel

class MyModelTrainer(BaseTrainer):
    model_config: MyModelConfig  # *중요: Type hint

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

        self.model = MyModel()

    def fit(self, train_dataset, valid_dataset):
        # X_tr, y_tr = self._dataset_to_numpy(train_dataset)
        # X_va, y_va = self._dataset_to_numpy(valid_dataset)
        tr_loader = self.make_loader(train_dataset, shuffle=True)
        for batch in tr_loader:
            x: torch.Tensor = batch["x"]
            y: torch.Tensor = batch["y"]
            sample_ids: list = batch["sample_ids"]
            metadata: list = batch["metadata"]

            print(type(x), type(y), type(sample_ids), type(metadata))
            print(x.shape)
            print(y.shape)
            exit()

        exit()

        # n_tasks = y_tr.shape[1]
        # output_dim = [4] * n_tasks

        # self.model = TabNetMultiTaskClassifier(
        #     output_dim=output_dim,
        #     device_name="cuda",
        # )
        # self.model.fit(
        #     X_tr, y_tr,
        #     eval_set=[(X_va, y_va)],
        #     eval_name=["valid"],
        #     eval_metric=["accuracy"],
        #     max_epochs=self.model_config.max_epochs,
        #     patience=self.model_config.patience_count,
        #     batch_size=self.model_config.batch_size,
        #     num_workers=self.model_config.num_workers,
        #     drop_last=self.model_config.drop_last,
        # )
        # return dict(self.model.history.history)

    def eval(self, test_dataset) -> dict[str, Any]:
        X_te, y_te = self._dataset_to_numpy(test_dataset)
        pred_list = self.model.predict(X_te)   # list[np.ndarray], len=T_out
        y_pred = np.stack([np.asarray(p).reshape(-1) for p in pred_list], axis=1).astype(np.int64)
        return compute_multitask_classification_metrics(y_te, y_pred)

    def save(self, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
        out = save_path / self.model_config.save_model_name
        self.model.save_model(str(out))  # save_model.zip
        return out

    def load(self, model_path: Path):
        self.model = TabNetMultiTaskClassifier(device_name="cuda")
        model_path = model_path / f"{self.model_config.save_model_name}.{self.model_config.model_ext}"
        self.model.load_model(str(model_path))
        return self.model
