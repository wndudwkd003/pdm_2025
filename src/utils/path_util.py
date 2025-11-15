from datetime import datetime
from pathlib import Path

def get_save_path(
    model_type: str,
    data_type: str,
    save_dir: str,
    masking_ratio: float,
    masking_mode: str
) -> Path:
    # 저장하는 방법 -> 모델타입_데이터타입_시간
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(save_dir) / f"{timestamp}_{model_type.value}_{data_type.value}_{masking_mode}_{masking_ratio}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir
