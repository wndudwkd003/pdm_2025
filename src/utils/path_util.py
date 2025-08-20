from datetime import datetime
from pathlib import Path

def get_save_path(
    model_type,
    data_type,
    save_dir,
) -> Path:
    # 저장하는 방법 -> 모델타입_데이터타입_시간
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(save_dir) / f"{model_type.value}_{data_type.value}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir
