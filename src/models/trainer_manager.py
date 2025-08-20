from configs.params.model import ModelType
from configs.maps.trainer_maps import TRAINER_MAP

class TrainerManager:
    @staticmethod
    def get_trainer(model_type: ModelType, **kwargs):

        if model_type not in TRAINER_MAP:
            raise ValueError(f"Trainer for model type {model_type} is not defined.")

        trainer_cls = TRAINER_MAP[model_type]()
        return trainer_cls(**kwargs)
