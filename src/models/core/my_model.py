
import torch
import torch.nn as nn
from typing import Sequence
from src.models.core.utils.groups import create_group_matrix


import warnings
warnings.filterwarnings("ignore", message="xFormers is available")


from src.models.core.module.sequences import (
    PositionalEmbedding,
    SequencePoolMean,
    TemporalWeightedMeanPool
)
from src.models.core.module.image_encoder import DINOv2EncoderHub
from src.models.core.module.fusions import FusionConcat

from src.models.core.module.head_module import Classifier
from src.models.core.module.tabular_encoder import TabNetEncoder
from src.models.core.module.category import EmbeddingGenerator

from src.models.core.encoder.mpie import MPIEncoder
from src.models.core.encoder.mpde import (
    MPDEncoder,
    # MPDDecoder
)
from src.models.core.module.dmattention import TransformerEncoderWithAttn
from deeptlf import DeepTFL, TreeDrivenEncoder


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
        cat_emb_dim: list = [], # 1 이상의 정수값이 원소로 들어가야 함
        d_model: int = 256,
        nhead: int = 8,
        ff_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        image_encoder_model: str = "dinov2_vits14",
        image_input_size: int = 224,
        multimodal_setting: bool = False,
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
        # self.d_model = d_model
        self.image_encoder_model = image_encoder_model
        self.image_input_size = image_input_size

        # grouped_features가 빈 리스트인경우 단위행렬(항등행렬)
        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        # 범주형 변수 임베딩 클래스``
        self.embedder = EmbeddingGenerator(
            input_dim=self.input_dim,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dims=self.cat_emb_dim,
            group_matrix=self.group_matrix,
        )
        self.post_embed_dim = self.embedder.post_embed_dim

        # Missing Pattern Independent Encoder (MPIE)
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

        # Missing Pattern Dependent Encoder (MPDE)
        # self.mpde = MPDEncoder(
        #     input_dim=self.post_embed_dim,
        #     n_d=self.n_d,
        #     n_a=self.n_a,
        #     n_shared=self.n_shared,
        #     n_independent=self.n_independent,
        #     n_steps=self.n_steps,
        #     virtual_batch_size=self.virtual_batch_size,
        #     momentum=self.momentum,
        #     group_attention_matrix=self.embedder.embedding_group_matrix,
        #     mask_type=self.mask_type,
        #     bias=self.bias,
        #     epsilon=self.epsilon,
        #     gamma=self.gamma
        # )

        # Missing Pattern Dependent Decoder (MPDD)
        # self.mpdd = MPDDecoder(
        #     input_dim=self.post_embed_dim,
        #     n_d=self.n_d,
        #     n_a=self.n_a,
        #     n_shared=self.n_shared,
        #     n_independent=self.n_independent,
        #     n_steps=self.n_steps,
        #     virtual_batch_size=self.virtual_batch_size,
        #     momentum=self.momentum,
        #     group_attention_matrix=self.embedder.embedding_group_matrix,
        #     mask_type=self.mask_type,
        #     bias=self.bias,
        #     epsilon=self.epsilon,
        #     gamma=self.gamma
        # )


        # 이미지 인코더
        if multimodal_setting:
            self.image_encoder = DINOv2EncoderHub(model_name=self.image_encoder_model)


        # 단순 컨캣 융합
        # self.fusion = FusionConcat(d_model=self.d_model)

        self.tab_proj = nn.Linear(self.n_d, self.n_d)
        self.activate = nn.LeakyReLU()

        # 포지셔널 임베딩 + Transformer 인코더
        self.pos_emb = PositionalEmbedding(max_seq_len=max_seq_len, d_model=self.n_d)

        self.transformer = TransformerEncoderWithAttn(
            num_layers=num_layers,
            d_model=self.n_d,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            norm_first=True,
        )

        # 시퀀스 풀링 + 멀티태스크 분류 헤드
        self.pool = TemporalWeightedMeanPool(init_alpha=1.0)
        self.classifier = Classifier(input_dim=self.n_d, output_dim=self.output_dim)


    def time_series_serialization(
        self,
        x: torch.Tensor,
    ):
        shape = x.shape
        L = len(shape)

        if L == 3: # (B, S, F)
            B, S, F = shape
            x_flat = x.view(B * S, F)

        return x_flat

    def time_series_grouping(
        self,
        x: torch.Tensor,
        shape: torch.Tensor,
    ):
        if len(shape) == 3: # (B, S, F)
            B, S, F = shape
            x_grouped = x.view(B, S, -1)

        return x_grouped


    """
    결측 값을 0으로 변환
    todo: 나중에 평균 또는 다른 방법으로 전환하는 세팅을 추가하면 어떨지
    """
    def _convert_zero_nan(self, x: torch.Tensor, x_bemv: torch.Tensor):
        missing_mask = (x_bemv == 0)
        x = x.clone()
        x[missing_mask] = 0.0
        return x


    def forward(
        self,

        # Tabular 입력 (B, S, F)
        x: torch.Tensor,

        # Binary Encoding of Missing Values (B, S, F)
        x_bemv: torch.Tensor,

        # todo: 멀티모달 세팅을 위해 이미지 입력을 추가할 수 있으나 나중에 고려 (B, S, C, H, W)
        x_img: torch.Tensor = None,
    ):
        # tabular
        B, S, F = x.shape

        x = self._convert_zero_nan(x, x_bemv)

        # tabular + time series serialization
        x_flat = self.time_series_serialization(x)
        x_bemv_flat = self.time_series_serialization(x_bemv)


        # 범주형 변수 포함 변환
        x_flat = self.embedder(x_flat)

        # MPDEncoder
        # step_outputs_mpde, M_loss_mpde, attention_maps_mpde = self.mpde(x_flat, x_bemv_flat)


        # tabular Missing Pattern Independence Encoder (MPIE)
        step_outputs, M_loss, attention_maps = self.mpie(x_flat, x_bemv_flat)

        # 여러 step을 합쳐서 최종 토큰 (TabNet-style)
        tab_tok = torch.sum(torch.stack(step_outputs, dim=0), dim=0)  # (B*S, n_d)

        # (B*S, n_d) → (B, S, n_d)
        tab_tok = self.time_series_grouping(tab_tok, x.shape)  # (B, S, n_d)

        # print("tab_tok.shape:", tab_tok.shape)

        # (B, S, n_d) → (B, S, n_d)
        z = self.tab_proj(tab_tok)
        z = self.activate(z)


        # 포지셔널 임베딩 + Transformer 인코더
        z = self.pos_emb(z)        # (B, S, n_d)
        z, tr_attn_maps = self.transformer(z)  # z: (B, S, n_d), tr_attn_maps: list[L] of (B, H, S, S)


        # 시퀀스 풀링 + 멀티태스크 classifier
        h = self.pool(z)       # (B, n_d)
        outs = self.classifier(h)  # List[(B, Ck)]

        # 학습에서 쓸 수 있도록 그대로 반환
        return outs, M_loss, z, attention_maps, tr_attn_maps




