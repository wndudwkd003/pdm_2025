from configs.params.model import ModelType

def get_node_model_config():
    from configs.model.model_configs import NODEConfig
    return NODEConfig


def get_tabnet_model_config():
    from configs.model.model_configs import TabNetConfig
    return TabNetConfig


def get_xgboost_model_config():
    from configs.model.model_configs import XGBConfig
    return XGBConfig


MODEL_CONFIG_MAP = {
    ModelType.NODE: get_node_model_config,
    ModelType.TABNET: get_tabnet_model_config,
    ModelType.XGBOOST: get_xgboost_model_config,
}

