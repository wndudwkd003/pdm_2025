# src/models/trainer_custom/mptms/xgb_trainer.py
from pathlib import Path
from typing import Any
import json
import numpy as np
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer


class XGBTrainer(BaseTrainer):
    def __init__(
        self,
        work_dir: Path,
        data_collator: BaseCollator,
        *,
        num_classes: int = 4,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        early_stopping_rounds: int = 50,
        num_workers: int = 4,
        collect_batch_size: int = 64,
        tree_method: str = "gpu_hist",
        predictor: str = "gpu_predictor",
        random_state: int = 42,
    ):
        super().__init__(work_dir=work_dir, data_collator=data_collator)
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.early_stopping_rounds = early_stopping_rounds
        self.num_workers = num_workers
        self.collect_batch_size = collect_batch_size
        self.tree_method = tree_method
        self.predictor = predictor
        self.random_state = random_state

        self.models: list[XGBClassifier] = []

    def _dataset_to_numpy(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            dataset,
            batch_size=self.collect_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
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

        T = y_tr.shape[1]
        self.models = []
        history: dict[str, Any] = {}

        for t in range(T):
            clf = XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                objective="multi:softprob",
                num_class=self.num_classes,
                tree_method=self.tree_method,
                predictor=self.predictor,
                n_jobs=self.num_workers,
                random_state=self.random_state,
                # ↓↓↓ v3에서는 생성자에 넣습니다
                eval_metric="mlogloss",
                early_stopping_rounds=self.early_stopping_rounds,
            )
            clf.fit(
                X_tr, y_tr[:, t],
                eval_set=[(X_tr, y_tr[:, t]), (X_va, y_va[:, t])],
                # eval_metric/early_stopping_rounds 인자는 제거
                # verbose 인자도 생략(원하시면 남겨도 무방)
            )
            self.models.append(clf)

            ev = clf.evals_result()
            history[f"task{t}_train_mlogloss"] = ev["validation_0"]["mlogloss"]
            history[f"task{t}_valid_mlogloss"]  = ev["validation_1"]["mlogloss"]
            history[f"task{t}_best_iteration"]  = int(clf.best_iteration)

        return history

    def eval(self, test_dataset) -> dict[str, Any]:
        X_te, y_te = self._dataset_to_numpy(test_dataset)   # (N,D), (N,T)
        T = y_te.shape[1]

        preds: list[np.ndarray] = []
        for t in range(T):
            p = self.models[t].predict(X_te)                # (N,)
            preds.append(p)
        y_pred = np.stack(preds, axis=1).astype(np.int64)   # (N, T)

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
        meta = {"num_tasks": len(self.models), "num_classes": self.num_classes}
        with open(save_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        for t, m in enumerate(self.models):
            m.save_model(str(save_path / f"save_model_task{t}.json"))
        return save_path

    def load(self, model_path: Path):
        with open(model_path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        num_tasks = int(meta["num_tasks"])
        self.models = []
        for t in range(num_tasks):
            clf = XGBClassifier()
            clf.load_model(str(model_path / f"save_model_task{t}.json"))
            self.models.append(clf)
        return self.models
