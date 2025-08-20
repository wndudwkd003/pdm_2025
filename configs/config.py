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

    model_type: ModelType = ModelType.TABNET

    save_dir: str = "outputs"
