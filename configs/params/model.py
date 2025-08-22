from enum import Enum

class ModelType(Enum):
    TABNET = "tabnet"
    MULTIMODAL_TABNET = "multimodal_tabnet"
    XGBOOST = "xgboost"
    NODE = "node"
    BASIC_TABULAR = "basic_tabular"
