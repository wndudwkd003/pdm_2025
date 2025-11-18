from configs.params.model import ModelType
from configs.maps.model_config_maps import MODEL_CONFIG_MAP

class ConfigManager:
    @staticmethod
    def get_model_config(model_type: ModelType):
        if model_type not in MODEL_CONFIG_MAP:
            raise ValueError(f"Model config for model type {model_type} is not defined.")

        model_config_cls = MODEL_CONFIG_MAP[model_type]()
        return model_config_cls()
