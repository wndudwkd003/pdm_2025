import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        S = x.size(1)
        return x + self.pos[:, :S, :]



class SequencePoolMean(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        return x.mean(dim=1)



class TemporalWeightedMeanPool(nn.Module):
    """
    마지막 타임스텝(현재 시점)에 가까울수록 가중치를 크게 주는
    temporal-weighted average pooling.

    - 거리 d_t = (S-1) - t 에 대해
      logit_t = - alpha * d_t
      w_t = softmax(logit_t)
    - alpha > 0 이므로 항상 최근일수록 가중치가 크고,
      과거로 갈수록 지수적으로 줄어듦.
    - alpha 는 학습 가능한 스칼라 파라미터.
    """

    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        # softplus(alpha_raw) > 0 이 되도록 raw 파라미터를 둠
        self.alpha_raw = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        B, S, D = x.shape
        device = x.device

        # 거리: 현재 시점(인덱스 S-1)에서 얼마나 떨어져 있는지
        # t: 0 ... S-1
        # d_t = (S-1) - t  => d_(S-1) = 0 (현재), d_0 = S-1 (가장 과거)
        positions = torch.arange(S, device=device)
        distances = (S - 1) - positions  # (S,)

        # alpha > 0 이 되도록 softplus 사용
        alpha = F.softplus(self.alpha_raw)  # 스칼라

        # logit_t = - alpha * d_t
        logits = -alpha * distances  # (S,)

        # softmax 로 정규화해서 가중치 합 = 1
        weights = torch.softmax(logits, dim=0)  # (S,)

        # (B, S, D) 에 브로드캐스트 가능하도록 reshape
        weights = weights.view(1, S, 1)        # (1, S, 1)

        # 가중합: sum_t w_t * x_t
        out = (x * weights).sum(dim=1)         # (B, D)
        return out
