import torch
import torch.nn as nn
import torch.nn.functional as F

embedding = nn.Embedding(10, 4)
weight = embedding.weight.clone().detach()

idx = torch.tensor([3])  # 인덱스 3

# Embedding으로 얻은 결과
out1 = embedding(idx)

# 원-핫 벡터 @ weight로 얻은 결과
one_hot = F.one_hot(idx, num_classes=10).float()
out2 = one_hot @ weight

print("nn.Embedding 결과:", out1)
print("원-핫 곱 결과:", out2)
print("차이:", (out1 - out2).abs().max())
