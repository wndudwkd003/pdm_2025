# Use the from-scratch Transformer above, but feed the concrete sensor table values from the user's diagram.
# We'll define x as (B=1, T=4, N=5) using the visible numeric columns:
# [Feat1, Feat2, Feat3, Feat_{n-1}, Feat_N] for each time step.
# No exception handling per user's instruction.

import torch
import torch.nn as nn
import math



torch.manual_seed(123)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):  # q,k,v: (B, H, T, D)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T_q, T_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)                              # (B, H, T_q, T_k)
        out = torch.matmul(attn, v)                                       # (B, H, T_q, D)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):  # (B, T, d_model) -> (B, H, T, head_dim)
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _merge(self, x):  # (B, H, T, head_dim) -> (B, T, d_model)
        B, H, T, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)
        return x

    def forward(self, x, mask=None):  # x: (B, T, d_model)
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        y, attn = self.attn(q, k, v, mask=mask)  # (B,H,T,D), (B,H,T,T)
        y = self._merge(y)                       # (B,T,d_model)
        y = self.o_proj(y)                       # (B,T,d_model)
        y = self.dropout(y)
        return y, attn

class PositionEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, B: int, T: int):
        pos = torch.arange(T).unsqueeze(0).expand(B, T)
        return self.emb(pos)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_mult * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_mult * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult, dropout)

    def forward(self, x, mask=None):
        h, attn = self.mha(self.ln1(x), mask=mask)
        x = x + h
        h = self.ffn(self.ln2(x))
        x = x + h
        return x, attn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, num_heads: int = 8,
                 num_layers: int = 2, ff_mult: int = 4, max_len: int = 256, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionEmbedding(max_len, d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ff_mult, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):  # x: (B, T, N)
        B, T, N = x.shape
        h = self.input_proj(x) + self.pos(B, T)  # (B, T, d_model)
        attn_maps = []
        for layer in self.layers:
            h, attn = layer(h, mask=mask)
            attn_maps.append(attn)
        return h, attn_maps

# ==== Build x from the diagram numbers ====
# Visible rows (T=4), columns (N=5): [Feat1, Feat2, Feat3, Feat_{n-1}, Feat_N]
rows = [
    [29.0, 22.0, 32.0, 48.99, 18.99],
    [29.0, 23.0, 31.0, 48.98, 18.98],
    [52.9, 37.0, 46.0, 49.89, 19.93],
    [52.97, 37.0, 45.0, 49.88, 19.81],
]

x = torch.tensor(rows, dtype=torch.float32).unsqueeze(0)  # (B=1, T=4, N=5)

model = TimeSeriesTransformer(n_features=5, d_model=64, num_heads=8, num_layers=2)

with torch.no_grad():
    y, attn = model(x)

print("Input x:", x.shape)
print("Encoder output y:", y.shape)
print("y[0]:\n", y[0])
print("\nLayer-1 head-1 attention (T x T):\n", attn[0][0, 0])
