from configs.params.model import ModelType
from configs.maps.trainer_maps import TRAINER_MAP
from src.data.collator_class.collator_base.base_collator import BaseCollator
from configs.model.model_configs import BaseModelConfig

class TrainerManager:
    @staticmethod
    def get_trainer(
        *,
        model_type: ModelType,
        work_dir: str,
        data_collator: BaseCollator,
        model_config: BaseModelConfig,
        metadata: dict = None,
        **kwargs
    ):
        if model_type not in TRAINER_MAP:
            raise ValueError(f"Trainer for model type {model_type} is not defined.")

        trainer_cls = TRAINER_MAP[model_type]()
        return trainer_cls(
            work_dir=work_dir,
            data_collator=data_collator,
            model_config=model_config,
            metadata=metadata,
            **kwargs
        )
