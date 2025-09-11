

from dataclasses import dataclass
from typing import Sequence

@dataclass
class BaseModelConfig:
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 30
    drop_last: bool = True
    pin_memory: bool = True
    max_epochs: int = 500
    learning_rate: float = 2e-2 # 1e-3
    weight_decay: float = 1e-5
    patience_count: int = 10
    num_classes: int = 4
    save_model_name: str = "save_model"



@dataclass
class MyModelConfig(BaseModelConfig):
    model_ext: str = "zip"




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

