from configs.params.data import DatasetType, DATASET_MAP

class DatasetManager:
    @staticmethod
    def get_class(data_type: DatasetType):
        if data_type not in DATASET_MAP:
            raise ValueError(f"Dataset type {data_type} is not registered.")

        return DATASET_MAP[data_type]

