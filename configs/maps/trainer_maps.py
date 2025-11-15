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

def get_tabnet_2_trainer():
    from src.models.trainer.custom_trainer.tabnet_2_trainer import TabNet2Trainer
    return TabNet2Trainer

def get_tabm_trainer():
    from src.models.trainer.custom_trainer.tabm_trainer import TabMTrainer
    return TabMTrainer


def get_my_model_trainer():
    from src.models.trainer.custom_trainer.my_trainer import MyModelTrainer
    return MyModelTrainer


TRAINER_MAP = {
    ModelType.TABNET: get_tabnet_trainer,
    ModelType.XGBOOST: get_xgb_trainer,
    ModelType.NODE: get_node_trainer,
    ModelType.TABNET_PYTABULAR: get_tabnet_2_trainer,
    ModelType.TABM: get_tabm_trainer,
    ModelType.MYMODEL: get_my_model_trainer,
}
