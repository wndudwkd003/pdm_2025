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
        horizon: int = 10,
        num_classes: int = 4,
        feature_hidden_dims: Sequence[int] = (64, 32, 32, 16, 8),
        nhead: int = 4,
        num_layers: int = 2,
        decoder_hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.num_classes = num_classes
        self.num_tokens = num_classes + 1
        self.start_idx = num_classes

        self.embedder = Embedder(embed_dim)

        self.mpie_model = MPIEModel(embed_dim=embed_dim, hidden_dims=feature_hidden_dims)
        self.mpid_model = MPIDModel(embed_dim=embed_dim, hidden_dims=feature_hidden_dims[::-1])

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

        self.latent_dim = input_dim * mpie_out_dim

        self.class_embed = nn.Embedding(self.num_tokens, decoder_hidden_dim)

        self.decoder = nn.GRU(
            input_size=decoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            batch_first=True,
        )

        self.decoder_out = nn.Linear(decoder_hidden_dim, num_classes)

        self.latent_to_hidden = nn.Linear(self.latent_dim, decoder_hidden_dim)

    def _convert_zero_nan(self, x: torch.Tensor, x_bemv: torch.Tensor):
        missing_mask = (x_bemv == 0)
        x = x.clone()
        x[missing_mask] = 0.0
        return x

    def forward(self, x, bemv):
        x_non_zero = self._convert_zero_nan(x, bemv)

        x_emb, bemv_emb = self.embedder(x_non_zero, bemv)

        x_feat = self.mpie_model(x_emb)

        h_list = []
        for i in range(self.input_dim):
            xf = x_feat[:, :, i, :]
            h = self.sequence_encoders[i](xf)
            h_list.append(h)

        h_all = torch.stack(h_list, dim=2)

        recon_emb = self.mpid_model(h_all)
        recon = self.output_proj(recon_emb).squeeze(-1)

        B = x.size(0)
        latent = h_all.mean(dim=1)
        latent = latent.reshape(B, -1)

        h_t = self.latent_to_hidden(latent).unsqueeze(0)

        logits_list = []

        y_prev = torch.full(
            (B,),
            self.start_idx,
            dtype=torch.long,
            device=latent.device,
        )

        for _ in range(self.horizon):
            dec_in = self.class_embed(y_prev).unsqueeze(1)
            out, h_t = self.decoder(dec_in, h_t)
            logits = self.decoder_out(out.squeeze(1))
            logits_list.append(logits)
            y_prev = torch.argmax(logits, dim=-1)

        preds = torch.stack(logits_list, dim=1)

        return {
            "latent": latent,
            "recon": recon,
            "preds": preds,
        }
