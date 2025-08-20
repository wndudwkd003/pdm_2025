import os, numpy as np, matplotlib.pyplot as plt

# ── 데이터 로드 ───────────────────────────────────
X_train = np.load("X_train.npy")      # (N_tr, 8, 24)
y_train = np.load("y_train.npy")
X_valid = np.load("X_valid.npy")      # (N_va, 8, 24)
y_valid = np.load("y_valid.npy")

WIN_LEN, N_FEAT = X_train.shape[1:3]  # 8, 24
np.random.seed(0)

# ── 센서별 Min-Max 산출 (train+valid 전체) ──────────
all_data = np.concatenate([X_train, X_valid], axis=0)        # (N_tot, 8, 24)
feat_min = all_data.min(axis=(0, 1))                         # (24,)
feat_max = all_data.max(axis=(0, 1))
feat_rng = np.where(feat_max - feat_min == 0,
                    1.0, feat_max - feat_min)                # 0-division 방지

def normalize(arr):
    # arr: (N, 8, 24) → 정규화 반환
    return (arr - feat_min) / feat_rng

X_train_norm = normalize(X_train)
X_valid_norm = normalize(X_valid)

# ── 시각화 & 저장 ────────────────────────────────
def save_plots(X, y, prefix):
    os.makedirs(f"plots/{prefix}", exist_ok=True)
    idx = np.random.choice(len(X), 10, replace=False)
    time_axis = range(WIN_LEN)
    for i, n in enumerate(idx):
        plt.figure(figsize=(6, 4))
        for f in range(N_FEAT):
            plt.plot(time_axis, X[n, :, f], linewidth=1)
        plt.title(f"{prefix} sample {i:02d}  label={y[n]}")
        plt.xlabel("time step"); plt.ylabel("normalized value (0-1)")
        plt.tight_layout()
        plt.savefig(f"plots/{prefix}/{prefix}_{i:02d}.png")
        plt.close()

save_plots(X_train_norm, y_train, "train")
save_plots(X_valid_norm, y_valid, "valid")
