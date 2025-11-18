# src/models/trainer_custom/mptms/xgb_trainer.py
from pathlib import Path
from typing import Any
import json
import numpy as np
from xgboost import XGBClassifier

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from src.utils.metrics import compute_multitask_classification_metrics
from configs.model.model_configs import XGBConfig

class XGBTrainer(BaseTrainer):
    model_config: XGBConfig  # *중요: Type hint

    def __init__(
        self,
        *,
        work_dir: Path,
        data_collator: BaseCollator,
        model_config: XGBConfig,
        metadata: dict[str, Any] = None
    ):
        super().__init__(
            work_dir=work_dir,
            data_collator=data_collator,
            model_config=model_config,
            metadata=metadata
        )

        self.models: list[XGBClassifier] = []

    def fit(self, train_dataset, valid_dataset):
        X_tr, y_tr = self._dataset_to_numpy(train_dataset)
        X_va, y_va = self._dataset_to_numpy(valid_dataset)

        print(X_tr.shape)


        T = y_tr.shape[1]
        self.models = []
        history: dict[str, Any] = {}

        for t in range(T):
            m = self.model_config.eval_metric

            clf = XGBClassifier(
                objective=self.model_config.objective,
                num_class=self.model_config.num_classes,
                random_state=self.model_config.seed,
                eval_metric=m,
                early_stopping_rounds= self.model_config.early_stopping_rounds,
            )
            clf.fit(
                X_tr, y_tr[:, t],
                eval_set=[(X_tr, y_tr[:, t]), (X_va, y_va[:, t])],
            )
            self.models.append(clf)

            ev = clf.evals_result()

            history[f"task{t}_train_{m}"] = ev["validation_0"][m]
            history[f"task{t}_valid_{m}"]  = ev["validation_1"][m]
            history[f"task{t}_best_iteration"]  = int(clf.best_iteration)

        return history

    def eval(self, test_dataset) -> dict[str, Any]:
        X_te, y_te = self._dataset_to_numpy(test_dataset)

        preds: list[np.ndarray] = []
        for t in range(y_te.shape[1]):
            print("Input and Output shape: ", X_te.shape, y_te.shape)
            p = self.models[t].predict(X_te)
            preds.append(p)
        y_pred = np.stack(preds, axis=1).astype(np.int64)
        return compute_multitask_classification_metrics(y_te, y_pred)

    def save(self, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
        meta = {"num_tasks": len(self.models)}
        with open(save_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        for t, m in enumerate(self.models):
            m.save_model(str(save_path / f"{self.model_config.save_model_name}{t}.{self.model_config.model_ext}"))
        return save_path

    def load(self, model_path: Path):
        with open(model_path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        num_tasks = int(meta["num_tasks"])
        self.models = []
        for t in range(num_tasks):
            clf = XGBClassifier()
            clf.load_model(str(model_path / f"{self.model_config.save_model_name}{t}.{self.model_config.model_ext}"))
            self.models.append(clf)
        return self.models
