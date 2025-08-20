from configs.params.data import DatasetType
from configs.maps.data_maps import DATASET_MAP

class DatasetManager:
    @staticmethod
    def get_class(data_type: DatasetType):
        if data_type not in DATASET_MAP:
            raise ValueError(f"Dataset type {data_type} is not registered.")

        data_cls = DATASET_MAP[data_type]()
        return data_cls

