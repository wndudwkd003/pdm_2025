from abc import ABC, abstractmethod
from typing import Any, Type
from pathlib import Path
from src.data.collator_class.collator_base.base_collator import BaseCollator

class BaseTrainer(ABC):
    """모든 트레이너의 기본 클래스"""

    def __init__(
        self,
        work_dir: Path,
        data_collator: BaseCollator,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.data_collator = data_collator

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
