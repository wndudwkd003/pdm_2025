from configs.params.data import DatasetType
from configs.params.model import ModelType
from configs.maps.collator_maps import COLLATOR_MAP

class CollatorManager:

    @classmethod  # 인스턴스 생성 없이 사용
    def get_collator(cls, data_type: DatasetType, model_type: ModelType):
        key = (data_type, model_type)

        if key not in COLLATOR_MAP:
            raise ValueError(f"지원하지 않는 데이터 타입 또는 모델 타입입니다: {key}")

        return COLLATOR_MAP[key]()

