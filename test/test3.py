import torch
import torch.nn as nn

# 랜덤 입력 준비
torch.manual_seed(42)
batch_size, input_dim, output_dim = 5, 16, 8
x = torch.randn(batch_size, 2 * output_dim)  # GLU 입력은 반드시 2*output_dim 차원

# 방법 1: 직접 구현
a = x[:, :output_dim]
b = x[:, output_dim:]
manual_out = a * torch.sigmoid(b)

# 방법 2: nn.GLU
glu = nn.GLU(dim=-1)
glu_out = glu(x)

# 비교
print("Manual out:\n", manual_out)
print("nn.GLU out:\n", glu_out)

print("\n두 출력 차이 (L2 norm):", torch.norm(manual_out - glu_out).item())
print("모든 요소 동일?:", torch.allclose(manual_out, glu_out, atol=1e-7))
