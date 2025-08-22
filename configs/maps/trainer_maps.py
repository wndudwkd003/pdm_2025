from configs.params.model import ModelType

def get_tabnet_trainer():
    from src.models.trainer.custom_trainer.tabnet_trainer import TabNetTrainer
    return TabNetTrainer


def get_xgb_trainer():
    from src.models.trainer.custom_trainer.xgboost_trainer import XGBTrainer
    return XGBTrainer

def get_node_trainer():
    from src.models.trainer.custom_trainer.node_trainer import NODETrainer
    return NODETrainer



TRAINER_MAP = {
    ModelType.TABNET: get_tabnet_trainer,
    ModelType.XGBOOST: get_xgb_trainer,
    ModelType.NODE: get_node_trainer,
}
