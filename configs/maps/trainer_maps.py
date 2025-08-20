from configs.params.model import ModelType

def get_tabnet_trainer():
    from src.models.trainer.custom_trainer.tabnet_trainer import TabNetTrainer
    return TabNetTrainer


def get_xgb_trainer():
    from src.models.trainer.custom_trainer.xgboost_trainer import XGBTrainer
    return XGBTrainer



TRAINER_MAP = {
    ModelType.TABNET: get_tabnet_trainer,
    ModelType.XGBOOST: get_xgb_trainer,
}
