import torch
import torch.nn as nn
from typing import Sequence
from src.models.core.module.input_adapter import Embedder
from src.models.core.mpie_w import MPIEModel, MPIDModel


class HybridDoubleBranchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        feature_hidden_dims: Sequence[int] = (64, 32, 32, 16, 8),
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embedder = Embedder(embed_dim)

        self.mpie_model = MPIEModel(
            embed_dim=embed_dim,
            hidden_dims=feature_hidden_dims
        )

        self.mpid_model = MPIDModel(
            embed_dim=embed_dim,
            hidden_dims=feature_hidden_dims[::-1]
        )

        mpie_out_dim = feature_hidden_dims[-1]

        self.sequence_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=mpie_out_dim,
                    nhead=nhead,
                    dim_feedforward=mpie_out_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                ),
                num_layers=num_layers,
            )
            for _ in range(input_dim)
        ])

        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        bemv: torch.Tensor
    ):
        x_emb, bemv_emb = self.embedder(x, bemv)

        x_feat = self.mpie_model(x_emb)

        h_list = []
        for i in range(self.input_dim):
            x_by_feat = x_feat[:, :, i, :]
            h = self.sequence_encoders[i](x_by_feat)
            h_list.append(h)

        h_all = torch.stack(h_list, dim=2)

        recon_emb = self.mpid_model(h_all)
        recon = self.output_proj(recon_emb).squeeze(-1)

        B = x.size(0)
        latent = h_all.mean(dim=1)
        latent = latent.view(B, -1)

        return {
            "latent": latent,
            "recon": recon,
        }
