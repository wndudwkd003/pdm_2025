from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from src.data.collator_class.collator_base.base_collator import BaseCollator
from configs.model.model_configs import BaseModelConfig


class BaseTrainer(ABC):
    def __init__(
        self,
        *,
        work_dir: Path,
        data_collator: BaseCollator,
        model_config: BaseModelConfig,
        metadata: dict[str, Any] = None
    ):
        self.model = None
        self.models = []
        self.work_dir = Path(work_dir)
        self.data_collator = data_collator
        self.model_config = model_config
        self.metadata = metadata


    def make_loader(
        self,
        dataset,
        shuffle: bool = False, # validation 시 False
    ):
        return DataLoader(
            dataset,
            batch_size=self.model_config.batch_size,
            shuffle=shuffle,
            num_workers=self.model_config.num_workers,
            collate_fn=self.data_collator,
            drop_last=self.model_config.drop_last,
            pin_memory=self.model_config.pin_memory,
        )

    def _dataset_to_numpy(
        self,
        dataset,
        shuffle: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        loader = self.make_loader(
            dataset,
            shuffle=shuffle
        )
        Xs: list[np.ndarray] = []
        Ys: list[np.ndarray] = []
        for batch in loader:
            Xs.append(batch["x"].cpu().numpy())   # (B, D)
            Ys.append(batch["y"].cpu().numpy())   # (B, T_out)
        X = np.concatenate(Xs, axis=0)            # (N, D)
        Y = np.concatenate(Ys, axis=0)            # (N, T_out)
        Y = Y.astype(np.int64)                    # 분류용 정수 레이블 가정
        return X, Y



    @abstractmethod
    def fit(
        self,
        train_dataset,
        valid_dataset
    ) -> dict[str, Any]:
        """모델 학습"""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def save(self, save_path: Path) -> Path:
        """모델 저장"""
        pass

    @abstractmethod
    def load(self, model_path: Path):
        """모델 로드"""
        pass
