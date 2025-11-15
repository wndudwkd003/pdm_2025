import torch
import torch.nn as nn

class PeriodicEmbedding(nn.Module):
    def __init__(self, k: int = 4, sigma: float = 0.1):
        super().__init__()
        # 주파수 파라미터 (trainable)
        self.c = nn.Parameter(torch.randn(k) * sigma)

    def forward(self, x):
        # x: (B,) 혹은 (B, 1) 형태의 숫자형 피처
        v = 2 * torch.pi * x.unsqueeze(-1) * self.c  # (B, k)
        sin_x = torch.sin(v)
        cos_x = torch.cos(v)
        return torch.cat([sin_x, cos_x], dim=-1)  # (B, 2k)

# 사용 예시
x = torch.tensor([0.1, 0.5, 1.0])  # 세 개 샘플
embed = PeriodicEmbedding(k=3, sigma=0.5)
out = embed(x)
print(out.shape)  # (3, 6)
print(out)
