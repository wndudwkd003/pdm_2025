from configs.params.model import ModelType

def get_tabnet_trainer():
    from src.models.trainer.custom_trainer.tabnet_trainer import TabNetTrainer
    return TabNetTrainer



TRAINER_MAP = {
    ModelType.TABNET: get_tabnet_trainer,
}
