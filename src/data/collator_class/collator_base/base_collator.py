# src/data/collator_class/collator_base/base_collator.py
from abc import ABC, abstractmethod
from typing import List, Any


class BaseCollator(ABC):
    """모든 콜레이터의 기본 클래스"""

    def __init__(self,
        append_mask_indicator: bool = True
    ):
        self.append_mask_indicator = append_mask_indicator
        pass

    @abstractmethod
    def __call__(self, batch: List[dict[str, Any]]) -> dict[str, Any]:
        """배치 데이터를 처리하여 모델 입력 형태로 변환"""
        pass


