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
        x_embed: torch.Tensor,           # (B, L)  or  (B, L, D)
        bemb: torch.Tensor,              # (B, L)  1=관측, 0=결측
        dense_mask: torch.Tensor | None = None
    ):
        # (0) 입력이 (B,L)이면 (B,L,1)로 확장
        squeeze_back = False
        if x_embed.dim() == 2:
            x_embed = x_embed.unsqueeze(-1)   # (B,L) -> (B,L,1)
            squeeze_back = True

        B, L, _ = x_embed.shape

        # --- 마스크 결합 ---
        # bemb: 1=관측, 0=결측
        # present: True=관측, False=결측
        use_mask = bemb
        present = use_mask > 0          # (B, L) bool

        # --- Q, K, V ---
        Q = self.Wq(x_embed).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,Dh)
        K = self.Wk(x_embed).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,Dh)
        V = self.Wv(x_embed).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,Dh)

        # --- 점곱 어텐션 ---
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,L,L)

        # (1) 열(col) 마스킹: 결측 feature를 key/value에서 제거
        col_mask = (~present).unsqueeze(1).unsqueeze(2).expand(B, self.n_heads, L, L)
        scores = scores.masked_fill(col_mask, float('-inf'))

        # softmax 후: 열 방향으로 결측 key/value에 대한 weight는 0이 됨
        A = torch.softmax(scores, dim=-1)  # (B,H,L,L)

        # (2) 행(row) 마스킹: 결측 feature를 query에서 제거
        row_mask = (~present).unsqueeze(1).unsqueeze(-1).expand(B, self.n_heads, L, L)
        A = A + row_mask.float() * -1.0
        A = torch.relu(A)  # 결측 row는 전부 음수였다가 0으로 깎임

        # --- 값 집계 ---
        out = torch.matmul(A, V)                             # (B,H,L,Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B,L,D)
        out = self.Wo(out)                                   # (B,L,D)

        # --- 원래 (B,L)이었으면 다시 (B,L) 로 ---
        if squeeze_back and self.d_model == 1:
            return out.squeeze(-1), A
        else:
            return out, A



class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.norm_first = norm_first

    def _sa_block(self, x: torch.Tensor):
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False,  # (B, num_heads, S, S) 형태
        )
        attn_output = self.dropout1(attn_output)
        return attn_output, attn_weights

    def _ff_block(self, x: torch.Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.dropout2(x)
        return x

    def forward(self, src: torch.Tensor):
        # src: (B, S, d_model)
        if self.norm_first:
            sa_out, attn = self._sa_block(self.norm1(src))
            src = src + sa_out
            ff_out = self._ff_block(self.norm2(src))
            src = src + ff_out
        else:
            sa_out, attn = self._sa_block(src)
            src = self.norm1(src + sa_out)
            ff_out = self._ff_block(src)
            src = self.norm2(src + ff_out)

        # src: (B, S, d_model)
        # attn: (B, num_heads, S, S)
        return src, attn


class TransformerEncoderWithAttn(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor):
        # src: (B, S, d_model)
        attn_maps: list[torch.Tensor] = []  # 각 레이어마다 (B, H, S, S)

        output = src
        for layer in self.layers:
            output, attn = layer(output)
            attn_maps.append(attn)

        output = self.norm(output)
        return output, attn_maps
