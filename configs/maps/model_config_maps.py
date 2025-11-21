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

def get_tabm_model_config():
    from configs.model.model_configs import TabMConfig
    return TabMConfig


def get_my_model_config():
    from configs.model.model_configs import MyModelConfig
    return MyModelConfig


MODEL_CONFIG_MAP = {
    ModelType.TABNET_PYTABULAR: get_node_model_config,
    ModelType.NODE: get_node_model_config,
    ModelType.TABNET: get_tabnet_model_config,
    ModelType.XGBOOST: get_xgboost_model_config,
    ModelType.TABM: get_tabm_model_config,
    ModelType.MYMODEL: get_my_model_config,
    ModelType.MYMODEL2: get_my_model_config,
}

