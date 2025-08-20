from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from src.data.collator_class.collator_base.base_collator import BaseCollator

class BaseTrainer:
    def __init__(
        self,
        work_dir: Path,
        data_collator,
        *,
        collect_batch_size: int = 64,
        num_workers: int = 4,
    ):
        self.work_dir = Path(work_dir)
        self.data_collator = data_collator
        self.collect_batch_size = collect_batch_size
        self.num_workers = num_workers

    def make_loader(self, dataset, *, batch_size: int | None = None, shuffle: bool = False):
        bs = self.collect_batch_size if batch_size is None else batch_size
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.data_collator,
            drop_last=False,
        )

    def dataset_to_numpy(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        loader = self.make_loader(dataset, batch_size=self.collect_batch_size, shuffle=False)
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
    def fit(self, train_dataset, valid_dataset=None) -> dict[str, Any]:
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
