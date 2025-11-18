# configs/config.py

from dataclasses import dataclass, field
from configs.params.data import DatasetType
from configs.params.model import ModelType

@dataclass
class Config:
    data_dir: str = "datasets/c-mapss/processed_data"
    data_type: DatasetType = DatasetType.CMPASS
    seed: int = 42

    train: str = "train"
    valid: str = "valid"
    test: str = "test"
    split_ratio: float = 0.2

    model_type: ModelType = ModelType.XGBOOST # MYMODEL

    save_dir: str = "outputs"

    # 마스킹 비율
    masking_ratio: float = 0.2


    # "none" | "mcar" | "block_t" | "per_sensor"
    masking_mode: str = "mcar"

    # True면 x 뒤에 마스크 인디케이터 붙임
    append_mask_indicator: bool = False

    # 마스킹 채움값
    mask_fill: float = -100.0

    csv_has_header: bool = False

    sincos_enabled: bool = True


    # === Test용 모델 로드 경로 ===
    load_model: str = "outputs/2025-11-18_12-16-45_xgboost_c-mapss_none_0.0"
