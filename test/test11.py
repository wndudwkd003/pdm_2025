import numpy as np
import matplotlib.pyplot as plt

# 파라미터 설정
m = 0.5  # decay rate
timesteps = 20

# 시간에 따른 weight 계산
i = np.arange(timesteps)
w = np.exp(-m * i)

# 시각화
plt.plot(i, w, marker="o")
plt.title(f"Temporal Ensemble Weights (m={m})")
plt.xlabel("i (time index)")
plt.ylabel("w_i")
plt.grid(True)
plt.show()
