from configs.params.data import DatasetType
from configs.params.model import ModelType


def get_mptms_tabnet_collator_class():
    from src.data.collator_class.collator_custom.mptms.tabnet_collator import MPTMSTabNetCollator
    return MPTMSTabNetCollator


COLLATOR_MAP = {
    (DatasetType.MPTMS, ModelType.TABNET): get_mptms_tabnet_collator_class,
    (DatasetType.MPTMS, ModelType.XGBOOST): get_mptms_tabnet_collator_class,
    (DatasetType.MPTMS, ModelType.NODE): get_mptms_tabnet_collator_class,
    (DatasetType.MPTMS, ModelType.TABNET_PYTABULAR): get_mptms_tabnet_collator_class,
    (DatasetType.MPTMS, ModelType.TABM): get_mptms_tabnet_collator_class,
    (DatasetType.MPTMS, ModelType.MYMODEL): get_mptms_tabnet_collator_class,
}
