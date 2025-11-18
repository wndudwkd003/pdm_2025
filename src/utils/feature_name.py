# src/utils/feature_names.py
from typing import Sequence, List

def expand_feature_names(
    sensors: Sequence[str],
    T: int,
    append_mask: bool
) -> List[str]:
    names = [f"{s}_t{t:02d}" for t in range(T) for s in sensors]      # 값 구간 (길이 F*T)
    if append_mask:
        names += [f"{s}_t{t:02d}_m" for t in range(T) for s in sensors] # 마스크 구간 (길이 F*T)
    return names
