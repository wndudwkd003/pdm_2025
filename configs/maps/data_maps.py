from configs.params.data import DatasetType


def get_mptms_dataset_class():
    from src.data.dataset_class.dataset_custom.mptms_dataset import MPTMSDataset
    return MPTMSDataset



DATASET_MAP = {
    DatasetType.MPTMS: get_mptms_dataset_class,
}
