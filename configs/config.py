# configs/config.py

from dataclasses import dataclass, field
from configs.params.data import DatasetType
from configs.params.model import ModelType

@dataclass
class Config:
    data_dir: str = "datasets/MPTMS/processed_data"
    data_type: DatasetType = DatasetType.MPTMS
    seed: int = 42

    train: str = "train"
    valid: str = "valid"
    test: str = "test"
    split_ratio: float = 0.2

    model_type: ModelType = ModelType.MYMODEL # MYMODEL

    save_dir: str = "outputs"

    # 마스킹 비율
    masking_ratio: float = 0.8


    # "none" | "mcar" | "block_t" | "per_sensor"
    masking_mode: str = "mcar"

    # True면 x 뒤에 마스크 인디케이터 붙임
    append_mask_indicator: bool = False

    # 마스킹 채움값
    mask_fill: float = -100.0

    csv_has_header: bool = True

    sincos_enabled: bool = True


    # === Test용 모델 로드 경로 ===
    load_model: str = "outputs/2025-11-18_10-07-05_mymodel_mptms_mcar_0.8"
