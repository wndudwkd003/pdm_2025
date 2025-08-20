from configs.params.data import DatasetType
from configs.params.model import ModelType


def get_mptms_tabnet_collator_class():
    from src.data.collator_class.collator_custom.mptms.tabnet_collator import MPTMSTabNetCollator
    return MPTMSTabNetCollator


COLLATOR_MAP = {
    (DatasetType.MPTMS, ModelType.TABNET): get_mptms_tabnet_collator_class,
}
