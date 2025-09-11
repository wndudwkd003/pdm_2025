import torch
import torch.nn as nn
import math


class MaskedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.Wo = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x_embed: torch.Tensor,           # (B, L)  또는  (B, L, D)
        bemb: torch.Tensor,              # (B, L)  존재(1)/결측(0)
        dense_mask: torch.Tensor | None = None  # (B, L) [선택] 연속 가중치(0~1)
    ):

        # scalar 입력이면 D 차원을 만들어줌.
        squeeze_back = False
        if x_embed.dim() == 2:
            # (B,L) → (B,L,1)
            x_embed = x_embed.unsqueeze(-1)
            squeeze_back = True

        B, L, _ = x_embed.shape


        # --- 마스크 결합 ---
        use_mask = bemb # if dense_mask is None else (bemb * dense_mask)  # (B, L)
        present = (use_mask > 0)                                         # (B, L) bool

        # --- Q, K, V ---
        Q = self.Wq(x_embed).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,Dh)
        K = self.Wk(x_embed).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,Dh)
        V = self.Wv(x_embed).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,Dh)


        # --- 점곱 어텐션 + 열(col) 마스킹 ---
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,L,L)
        col_mask = (~present).unsqueeze(1).unsqueeze(2).expand(B, self.n_heads, L, L)
        scores = scores.masked_fill(col_mask, float('-inf'))

        # --- softmax ---
        A = torch.softmax(scores, dim=-1)  # (B,H,L,L)

        # --- 행(row) 마스킹: Mᵀ을 더하고 ReLU ---
        row_mask = (~present).unsqueeze(1).unsqueeze(-1).expand(B, self.n_heads, L, L)
        # row_mask.bool() → True인 위치는 결측이므로 -1.0을 더해준다.
        A = A + row_mask.float() * -1.0
        A = torch.relu(A)  # 음수가 된 위치는 0으로 깎임

        # --- 값 집계 ---
        out = torch.matmul(A, V)                             # (B,H,L,Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B,L,D)
        out = self.Wo(out)                                   # (B,L,D)


        # --- 입력이 (B,L)이었으면 다시 (B,L)로 되돌림 ---
        if squeeze_back and self.d_model == 1:
            return out.squeeze(-1), A
        else:
            return out, A
# import torch
# import torch.nn as nn
# from xformers.ops import memory_efficient_attention

# class MaskedSelfAttention(nn.Module):
#     def __init__(self, d_model: int, n_heads: int):
#         super().__init__()
#         assert d_model % n_heads == 0
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_head = d_model // n_heads
#         self.Wq = nn.Linear(d_model, d_model, bias=True)
#         self.Wk = nn.Linear(d_model, d_model, bias=True)
#         self.Wv = nn.Linear(d_model, d_model, bias=True)
#         self.Wo = nn.Linear(d_model, d_model, bias=True)

#     def forward(
#         self,
#         x_embed: torch.Tensor,   # (B,L) 또는 (B,L,D) ; 마지막 차원은 d_model과 같아야 함
#         bemb: torch.Tensor,      # (B,L) ; 1=존재, 0=결측
#         dense_mask: torch.Tensor | None = None,
#     ):
#         # (B,L) → (B,L,1) : d_model=1, n_heads=1 구성에서만 사용
#         squeeze_back = False
#         if x_embed.dim() == 2:
#             x_embed = x_embed.unsqueeze(-1)  # (B,L,1)
#             squeeze_back = True

#         B, L, D = x_embed.shape
#         H = self.n_heads

#         # 마지막 차원은 반드시 d_model
#         assert D == self.d_model


#         Q = self.Wq(x_embed).view(B, L, H, self.d_head).transpose(1, 2).reshape(B*H, L, self.d_head)
#         K = self.Wk(x_embed).view(B, L, H, self.d_head).transpose(1, 2).reshape(B*H, L, self.d_head)
#         V = self.Wv(x_embed).view(B, L, H, self.d_head).transpose(1, 2).reshape(B*H, L, self.d_head)
#         print("Q shape:", Q.shape)
#         print("K shape:", K.shape)
#         print("V shape:", V.shape)


#         use_mask = bemb
#         print("use_mask:", use_mask.shape)

#         # use_mask: (B*H, L) 로 넘기세요. (이미 그렇게 쓰고 계신 모양새)
#         present  = (use_mask > 0)                      # (B*H, L) bool
#         key_mask = torch.logical_not(present)          # (B*H, L) bool

#         # 결측 key=-inf, 관측 key=0.0  → (B*H, L)
#         key_bias = torch.where(
#             key_mask, torch.full_like(use_mask, float("-inf"), dtype=torch.float32),
#             torch.zeros_like(use_mask, dtype=torch.float32)
#         )                                              # (B*H, L)

#         # ★ 여기! 3D로 만드세요. (B*H, L, L)
#         attn_bias = key_bias[:, :, None].expand(key_bias.shape[0], key_bias.shape[1], key_bias.shape[1])
#         # attn_bias: (B*H, L, L)

#         # xFormers 호출 (출력: (B*H, L, Dh))
#         out = memory_efficient_attention(Q, K, V, attn_bias=attn_bias)
#         print("out before row mask:", out.shape)
#         exit()

#         # --- 행(row) 마스킹: 결측 query(i)는 최종 출력 0 ---
#         present = (use_mask > 0).to(out.dtype)                   # (B,L)
#         present = present.unsqueeze(1).expand(B, H, L)           # (B,H,L)
#         present = present.reshape(B*H, L, 1)                     # (B*H,L,1)
#         out = out * present                                      # (B*H,L,Dh)

#         # (B*H,L,Dh) -> (B,H,L,Dh) -> (B,L,D)
#         out = out.view(B, H, L, self.d_head).transpose(1, 2).contiguous().view(B, L, D)
#         out = self.Wo(out)                                       # (B,L,D)

#         # 입력이 (B,L)이고 d_model=1이면 (B,L)로 되돌림
#         if squeeze_back and self.d_model == 1:
#             out = out.squeeze(-1)

#         return out
