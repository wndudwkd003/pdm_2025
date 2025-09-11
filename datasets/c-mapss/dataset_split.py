import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ── 설정 ─────────────────────────────────────────
DATA_PATH  = "datasets/nasa_turbofan/all_engines_labeled.csv"
WINDOW_LEN = 8          # 타임스텝 길이
STRIDE     = 1          # 슬라이딩 간격
TRAIN_RATE = 0.9        # train : valid = 9 : 1
RNG_SEED   = 42

# ── 데이터 로드 ───────────────────────────────────
df = pd.read_csv(DATA_PATH)

# 피처 24개만 추출 (dataset/unit/cycle/state 제외)
feature_cols = [c for c in df.columns
                if c not in ("dataset", "unit", "cycle", "state")]

Xs, ys, grp_keys = [], [], []          # 윈도우, 레이블, 그룹 키

# ── 윈도우 & 레이블 생성 ─────────────────────────
for (ds, uid), g in df.groupby(["dataset", "unit"], sort=False):
    g  = g.sort_values("cycle")                       # cycle 오름차순
    Xm = g[feature_cols].to_numpy(np.float32)         # (전체주기, 24)
    ym = g["state"].to_numpy(np.int64)

    # 슬라이딩 (t-WIN_LEN+1 … t) → y_{t+1}
    max_start = len(g) - WINDOW_LEN - 1
    for start in range(0, max_start + 1, STRIDE):
        end = start + WINDOW_LEN
        Xs.append(Xm[start:end])                      # (8, 24)
        ys.append(ym[end])                            # 한 스텝 앞(state)
        grp_keys.append(f"{ds}_{uid}")                # FD00x_unitID

X_all   = np.stack(Xs)                                # (N, 8, 24)
y_all   = np.asarray(ys, dtype=np.int64)
grp_all = np.asarray(grp_keys)

# ── 엔진 단위 9:1 분할 ───────────────────────────
gss = GroupShuffleSplit(n_splits=1,
                        train_size=TRAIN_RATE,
                        random_state=RNG_SEED)
tr_idx, va_idx = next(gss.split(X_all, y_all, groups=grp_all))

X_train, y_train = X_all[tr_idx], y_all[tr_idx]
X_valid, y_valid = X_all[va_idx], y_all[va_idx]
grp_train, grp_valid = grp_all[tr_idx], grp_all[va_idx]

# ── 중복 검증 ────────────────────────────────────
duplicated = np.intersect1d(grp_train, grp_valid)
if duplicated.size:
    raise RuntimeError(f"중복 그룹 발견 ⇒ {duplicated[:10]} ...")
print("검증 완료: train/valid 엔진 ID가 겹치지 않습니다.")

# ── NumPy 저장 ───────────────────────────────────
np.save("X_train.npy", X_train)   # (N_tr, 8, 24)
np.save("y_train.npy", y_train)   # (N_tr,)
np.save("X_valid.npy", X_valid)   # (N_va, 8, 24)
np.save("y_valid.npy", y_valid)   # (N_va,)

print("저장 완료:",
      f"X_train {X_train.shape}, y_train {y_train.shape}",
      f"X_valid {X_valid.shape}, y_valid {y_valid.shape}", sep="\n")
