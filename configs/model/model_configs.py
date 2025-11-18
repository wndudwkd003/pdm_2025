

from dataclasses import dataclass
from typing import Sequence

@dataclass
class BaseModelConfig:
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 3
    drop_last: bool = True
    pin_memory: bool = True
    max_epochs: int = 50
    learning_rate: float = 1e-3 # 1e-3
    weight_decay: float = 1e-5
    patience_count: int = 15
    num_classes: int = 4
    save_model_name: str = "save_model"
    multimodal_setting: bool = False



@dataclass
class MyModelConfig(BaseModelConfig):

    multimodal_setting: bool = False
    device: str = "cuda"
    model_ext: str = "zip"

    # 멀티태스크 출력 차원 (예: 미래 10스텝 * 클래스 4개)
    output_dims: Sequence[int] = (4,) * 10


    min_learning_rate: float = 1e-5
    lambda_sparse: float = 1e-3
    lambda_recon: float = 1.5


    # TabNet / MPIE / MPDE 관련 하이퍼파라미터
    n_d: int = 64
    n_a: int = 64
    n_shared: int = 4
    n_independent: int = 4
    n_steps: int = 8
    virtual_batch_size: int = 128
    momentum: float = 0.02
    mask_type: str = "sparsemax"
    bias: bool = True
    epsilon: float = 1e-6
    gamma: float = 1.0

    # Transformer 관련
    d_model: int = 256
    nhead: int = 8
    ff_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 256

    # 예: {"machine_id": {"cardinality": 32, "emb_dim": 16}, ...}
    cat_feature_info: dict[str, dict[str, int]] = None

    # 예: [["NTC", "PM1.0"], ["PM2.5", "PM10", "CT1"]]
    grouped_feature_names: list[list[str]] = None






@dataclass
class TabNetConfig(BaseModelConfig):
    model_ext: str = "zip"


@dataclass
class XGBConfig(BaseModelConfig):
    tree_method: str = "hist"
    early_stopping_rounds: int = BaseModelConfig.patience_count
    eval_metric: str = "mlogloss"
    model_ext: str = "json"
    objective: str = "multi:softprob"



@dataclass
class NODEConfig(BaseModelConfig):
    continuous_cols: Sequence[str] = ()
    categorical_cols: Sequence[str] = ()
    target_names: Sequence[str] = ()
    early_stopping: str = "valid_loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = BaseModelConfig.patience_count
    checkpoints: str = "valid_loss"
    load_best: bool = True
    auto_lr_find: bool = True


@dataclass
class TabMConfig(BaseModelConfig):
    k: int = 32
    model_ext: str = "pt"

