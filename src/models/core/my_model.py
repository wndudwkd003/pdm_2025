import torch
import torch.nn as nn

from typing import Sequence

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

from src.models.core.encoder.mpie import MPIEncoder
from src.models.core.encoder.mpde import (
    MPDEncoder,
    MPDDecoder,
)

from src.models.core.module.image_encoder import DINOv2EncoderHub
from src.models.core.module.fusions import FusionConcat
from src.models.core.module.category import EmbeddingGenerator
from src.models.core.module.head_module import Classifier
from src.models.core.module.dmattention import TransformerEncoderWithAttn

from src.models.core.module.sequences import (
    PositionalEmbedding,
    SequencePoolMean,
    TemporalWeightedMeanPool
)

from src.models.core.utils.groups import create_group_matrix

class MyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        output_dim: Sequence[int] = (4,),
        n_d: int = 256,
        n_a: int = 256,
        n_shared: int = 8,
        n_independent: int = 8,
        n_steps: int = 3,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        grouped_features: list[list[int]] = [],
        mask_type: str = "sparsemax",
        bias: bool = True,
        epsilon: float = 1e-6,
        gamma: float = 1.3,
        cat_idxs: list = [],
        cat_dims: list = [],
        cat_emb_dim: list = [],
        nhead: int = 8,
        ff_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        image_encoder_model: str = "dinov2_vits14",
        image_input_size: int = 224,
        multimodal_setting: bool = False,
        image_feat_dim: int = 384,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.n_steps = n_steps
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.grouped_features = grouped_features
        self.mask_type = mask_type
        self.bias = bias
        self.epsilon = epsilon
        self.gamma = gamma
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.image_encoder_model = image_encoder_model
        self.image_input_size = image_input_size
        self.multimodal_setting = multimodal_setting
        self.image_feat_dim = image_feat_dim

        self.dropout_branch = nn.Dropout(p=dropout)

        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        self.embedder = EmbeddingGenerator(
            input_dim=self.input_dim,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dims=self.cat_emb_dim,
            group_matrix=self.group_matrix,
        )
        self.post_embed_dim = self.embedder.post_embed_dim

        self.mpie = MPIEncoder(
            input_dim=self.post_embed_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_shared=self.n_shared,
            n_independent=self.n_independent,
            n_steps=self.n_steps,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            group_attention_matrix=self.embedder.embedding_group_matrix,
            mask_type=self.mask_type,
            bias=self.bias,
            epsilon=self.epsilon,
            gamma=self.gamma
        )

        self.mpde = MPDEncoder(
            input_dim=self.post_embed_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_shared=self.n_shared,
            n_independent=self.n_independent,
            n_steps=self.n_steps,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            group_attention_matrix=self.embedder.embedding_group_matrix,
            mask_type=self.mask_type,
            bias=self.bias,
            epsilon=self.epsilon,
            gamma=self.gamma
        )

        if self.multimodal_setting:
            # 1) DINOv2 freeze 모드
            self.image_encoder = DINOv2EncoderHub(
                model_name=self.image_encoder_model,
                freeze_backbone=True,
            )

            # 2) DINO feature (image_feat_dim) → n_d 로 가는 MLP head
            mlp_hidden_dim = self.n_d * 2
            self.image_mlp = nn.Sequential(
                nn.Linear(self.image_feat_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, self.n_d),
            )

        self.fuse_mpie_mpde = nn.Linear(2 * self.n_d, self.n_d)

        self.mpie_proj = nn.Linear(self.n_d, self.n_d)
        self.mpde_proj = nn.Linear(self.n_d, self.n_d)
        self.activate = nn.LeakyReLU()

        self.pos_emb = PositionalEmbedding(max_seq_len=max_seq_len, d_model=self.n_d)

        self.mpie_transformer = TransformerEncoderWithAttn(
            num_layers=num_layers,
            d_model=self.n_d,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            norm_first=True,
        )
        self.mpde_transformer = TransformerEncoderWithAttn(
            num_layers=num_layers,
            d_model=self.n_d,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            norm_first=True,
        )

        self.pool_mpie = TemporalWeightedMeanPool(init_alpha=1.0)
        self.pool_mpde = TemporalWeightedMeanPool(init_alpha=1.0)
        self.pool_img = TemporalWeightedMeanPool(init_alpha=1.0)

        self.branch_attn = nn.MultiheadAttention(
            embed_dim=self.n_d,
            num_heads=nhead,
            batch_first=True,
        )

        self.num_branches = 2 + (1 if self.multimodal_setting else 0)

        self.before_classifier = nn.Linear(self.n_d * self.num_branches, self.n_d)
        self.before_classifier_2 = nn.Linear(self.n_d, self.n_d)
        self.classifier = Classifier(input_dim=self.n_d, output_dim=self.output_dim)

    def time_series_serialization(
        self,
        x: torch.Tensor,
    ):
        shape = x.shape
        L = len(shape)

        if L == 3:
            B, S, F = shape
            x_flat = x.view(B * S, F)
            return x_flat

        return x

    def time_series_grouping(
        self,
        x: torch.Tensor,
        shape: tuple,
    ):
        B, S, F = shape
        x_grouped = x.view(B, S, -1)
        return x_grouped


    def _convert_zero_nan(self, x: torch.Tensor, x_bemv: torch.Tensor):
        missing_mask = (x_bemv == 0)
        x = x.clone()
        x[missing_mask] = 0.0
        return x

    def forward(
        self,
        x: torch.Tensor,
        x_bemv: torch.Tensor,
        x_img: torch.Tensor = None
    ):
        # 원본 x의 shape
        x_shape = x.shape

        # x_bemv에 해당하는 결측치를 x에 0으로 대체
        x = self._convert_zero_nan(x, x_bemv)

        # 시계열 데이터를 B*S, F 형태로 변환
        x_flat = self.time_series_serialization(x)
        x_bemv_flat = self.time_series_serialization(x_bemv)

        x_flat = self.embedder(x_flat)

        step_outputs_mpde, M_loss_mpde, att_dep = self.mpde(x_flat, x_bemv_flat)
        mpde_feat_flat = torch.sum(torch.stack(step_outputs_mpde, dim=0), dim=0)
        mpde_feat = self.time_series_grouping(mpde_feat_flat, x_shape)


        step_outputs_mpie, M_loss_mpie, mpie_attention_maps = self.mpie(x_flat, x_bemv_flat)
        mpie_feat_flat = torch.sum(torch.stack(step_outputs_mpie, dim=0), dim=0)
        mpie_feat = self.time_series_grouping(mpie_feat_flat, x_shape)


        z_mpie = self.mpie_proj(mpie_feat)
        z_mpie = self.activate(z_mpie)
        z_mpie, tr_attn_mpie = self.mpie_transformer(z_mpie)
        h_mpie = self.pool_mpie(z_mpie)

        z_mpde = self.mpde_proj(mpde_feat)
        z_mpde = self.activate(z_mpde)
        z_mpde, tr_attn_mpde = self.mpde_transformer(z_mpde)
        h_mpde = self.pool_mpde(z_mpde)

        h_mpie = self.dropout_branch(h_mpie)
        h_mpde = self.dropout_branch(h_mpde)

        branch_tokens = [h_mpie, h_mpde]

        if self.multimodal_setting:
            B_img, S_img, C, H_img, W_img = x_img.shape
            x_img_flat = x_img.view(B_img * S_img, C, H_img, W_img)

            # 1) frozen DINOv2 backbone에서 feature 추출 (gradient X)
            img_feat_flat = self.image_encoder(x_img_flat)            # (B*S, image_feat_dim)

            # 2) time dimension 복원
            img_feat_seq = img_feat_flat.view(B_img, S_img, -1)       # (B, S, image_feat_dim)

            # 3) temporal pooling (여기까지는 여전히 frozen feature)
            img_feat = self.pool_img(img_feat_seq)                    # (B, image_feat_dim)

            # 4) 학습되는 MLP head
            img_feat = self.image_mlp(img_feat)                       # (B, n_d)

            # 5) 나머지 branch들과 동일하게 활성화 후 사용
            img_feat = self.activate(img_feat)
            branch_tokens.append(img_feat)

        branch_seq = torch.stack(branch_tokens, dim=1)
        branch_out, branch_attn = self.branch_attn(branch_seq, branch_seq, branch_seq)


        B_out, T_out, D_out = branch_out.shape
        h_final = branch_out.reshape(B_out, T_out * D_out)


        h_final = self.before_classifier(h_final)
        h_final = self.activate(h_final)
        h_final = self.before_classifier_2(h_final)

        latent = self.activate(h_final)

        outs = self.classifier(latent)

        M_loss_total = M_loss_mpie + M_loss_mpde

        results = {
            "latent": latent,
            "outs": outs,
            "M_loss_total": M_loss_total,
        }

        return results
