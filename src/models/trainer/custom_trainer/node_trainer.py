# src/models/trainer_custom/mptms/basic_tabular_trainer.py
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from pytorch_tabular import TabularModel
from pytorch_tabular.models import NodeConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from src.utils.metrics import compute_multitask_classification_metrics

from configs.model.model_configs import NODEConfig
from src.data.collator_class.collator_custom.mptms.tabnet_collator import BaseCollator

from src.utils.feature_name import expand_feature_names
from pytorch_tabular.models.common.heads import LinearHeadConfig


class NODETrainer(BaseTrainer):
    model_config: NODEConfig # *중요: Type hint

    def __init__(
        self,
        *,
        work_dir: Path,
        data_collator: BaseCollator,
        model_config: NODEConfig,
        metadata: dict[str, Any] = None
    ):
        super().__init__(
            work_dir=work_dir,
            data_collator=data_collator,
            model_config=model_config,
            metadata=metadata
        )

        # 메타데이터 설정 (절대 get 메서드 사용하지 않음 -> 예기치 못한 에러를 그냥 넘어가는 버그 방지)
        self.model_config.continuous_cols = self.metadata["continuous_cols"]
        self.model_config.categorical_cols = self.metadata["categorical_cols"]
        self.model_config.target_names = self.metadata["target_names"]

        # 평탄화된 실제 피처명(D개) 생성
        self.step_feature_names = expand_feature_names(
            sensors=self.metadata["continuous_cols"],
            T=self.metadata["forward"],
            append_mask=self.data_collator.append_mask_indicator,
        )


    def _to_dataframe(self, X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
        N, D = X.shape
        F = len(self.metadata["continuous_cols"])
        T = int(self.metadata["forward"])
        append_mask = bool(self.data_collator.append_mask_indicator)
        D_expected = F * T * (2 if append_mask else 1)

        assert D == D_expected, (
            f"D mismatch: got {D}, expected {D_expected} "
            f"(F={F}, T={T}, append_mask={append_mask})"
        )
        assert len(self.step_feature_names) == D, "feature name count mismatch"

        dfX = pd.DataFrame(X, columns=self.step_feature_names)
        dfY = pd.DataFrame(Y, columns=self.model_config.target_names).astype("int64")
        return pd.concat([dfX, dfY], axis=1)



    def fit(
        self,
        train_dataset,
        valid_dataset
    ) -> dict[str, Any]:
        X_tr, Y_tr = self._dataset_to_numpy(train_dataset, shuffle=True)
        df_train = self._to_dataframe(X_tr, Y_tr)

        X_va, Y_va = self._dataset_to_numpy(valid_dataset, shuffle=False)
        df_valid = self._to_dataframe(X_va, Y_va)


        data_config = DataConfig(
            target=self.model_config.target_names,
            continuous_cols=self.step_feature_names,
            categorical_cols=self.model_config.categorical_cols,
            num_workers= self.model_config.num_workers,
        )


        model_cfg = NodeConfig(
            task="classification",
            metrics=["accuracy"], # ["f1_score", "accuracy", "auroc"],
            metrics_prob_input=[False], # [False, False, True],   # f1/acc는 라벨, auroc은 proba
            metrics_params=[
                # {"average": "macro", "num_classes": self.model_config.num_classes},  # f1_score
                {},                                                                  # accuracy
                # {"average": "macro", "num_classes": self.model_config.num_classes},  # auroc
            ],
            learning_rate=self.model_config.learning_rate,
        )

        trainer_config = TrainerConfig(
            max_epochs=self.model_config.max_epochs,
            batch_size=self.model_config.batch_size,
            early_stopping=self.model_config.early_stopping,
            early_stopping_patience=self.model_config.early_stopping_patience,
            early_stopping_mode=self.model_config.early_stopping_mode,
            auto_lr_find=False,
            checkpoints_path=str(self.work_dir / "checkpoints"),
            trainer_kwargs={
                "default_root_dir": str(self.work_dir)
            },
        )

        optimizer_config = OptimizerConfig(
            optimizer="torch.optim.Adam",
            # lr_scheduler="CosineAnnealingWarmRestarts",
            # lr_scheduler_params={"T_0": 10, "T_mult": 1, "eta_min": 1e-5},
        )

        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_cfg,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        self.tabular_model.fit(train=df_train, validation=df_valid)

        return {
            "n_train": len(df_train),
            "n_valid": len(df_valid),
            "features": self.step_feature_names,
            "targets": self.model_config.target_names,
        }


    def eval(self, test_dataset) -> dict[str, Any]:
        X_te, Y_te = self._dataset_to_numpy(test_dataset, shuffle=False)
        df_test = self._to_dataframe(X_te, Y_te)
        pred_df = self.tabular_model.predict(df_test)

        # 타깃별 예측 클래스 취합
        yhat_cols: list[np.ndarray] = []
        for tgt in self.model_config.target_names:
            pred_col = f"{tgt}_prediction"
            if pred_col in pred_df.columns:
                yhat_cols.append(pred_df[pred_col].to_numpy(dtype=np.int64))
            else:
                proba_cols = [c for c in pred_df.columns if c.startswith(f"{tgt}_proba_")]
                proba_cols = sorted(proba_cols, key=lambda c: int(c.split("_")[-1]))
                proba = pred_df[proba_cols].to_numpy(dtype=np.float32)
                yhat_cols.append(np.argmax(proba, axis=1).astype(np.int64))

        y_pred = np.stack(yhat_cols, axis=1)  # (N, T_out)
        return compute_multitask_classification_metrics(Y_te, y_pred)


    def save(self, save_path: Path) -> Path:
        save_path.mkdir(parents=True, exist_ok=True)
        self.tabular_model.save_model(str(save_path))
        return save_path

    def load(self, model_path: Path):
        self.tabular_model = TabularModel.load_model(str(model_path))
        return self.tabular_model
