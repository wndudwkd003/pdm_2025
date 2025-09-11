
import torch
import torch.nn as nn
from typing import Sequence as sequ
from src.models.core.utils.groups import create_group_matrix


import warnings
warnings.filterwarnings("ignore", message="xFormers is available")


from src.models.core.module.sequences import (
    PositionalEmbedding,
    SequencePoolMean,
)
from src.models.core.module.image_encoder import DINOv2EncoderHub
from src.models.core.module.fusions import FusionConcat

from src.models.core.module.head_module import Classifier
from src.models.core.module.tabular_encoder import TabNetEncoder
from src.models.core.module.category import EmbeddingGenerator

from src.models.core.encoder.mpie import MPIEncoder
from src.models.core.encoder.mpde import (
    MPDEncoder,
    MPDDecoder
)



class MyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        output_dim: sequ[int] = (4,),
        n_d: int = 64,
        n_a: int = 64,
        n_shared: int = 2,
        n_independent: int = 2,
        n_steps: int = 3,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        grouped_features: list[list[int]] = [],
        mask_type: str = "sparsemax",
        bias: bool = True,
        epsilon: float = 1e-6,
        gamma: float = 1.0,
        cat_idxs: list = [],
        cat_dims: list = [],
        cat_emb_dim: int|list = 1,
        d_model: int = 256,
        nhead: int = 8,
        ff_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        image_encoder_model: str = "dinov2_vits14",
        image_input_size: int = 224,
        multimodal_setting: bool = False
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
        self.d_model = 64+384
        self.image_encoder_model = image_encoder_model
        self.image_input_size = image_input_size
        # grouped_features가 빈 리스트인경우 단위행렬(항등행렬)
        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)
        print(self.group_matrix.shape)
        print(self.group_matrix)
        # 범주형 변수 임베딩 클래스
        self.embedder = EmbeddingGenerator(
            input_dim=self.input_dim,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dims=self.cat_emb_dim,
            group_matrix=self.group_matrix
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

        # Missing Pattern Dependent Decoder (MPDD)
        self.mpdd = MPDDecoder(
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

        # TabNet
        self.tabular_encoder = TabNetEncoder(
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
        # 이미지 인코더
        if multimodal_setting:
            self.image_encoder = DINOv2EncoderHub(model_name=self.image_encoder_model)




        # 단순 컨캣 융합
        self.fusion = FusionConcat(d_model=self.d_model)
        # 포지셔널 임베딩 + Transformer 인코더
        self.pos_emb = PositionalEmbedding(max_seq_len=max_seq_len, d_model=self.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # 시퀀스 풀링 + 멀티태스크 분류 헤드
        self.seq_pool = SequencePoolMean()
        self.classifier = Classifier(input_dim=self.d_model, output_dim=self.output_dim)


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


    def forward(
        self,
        x: torch.Tensor,      # (B, S, F)
        x_bemv: torch.Tensor,  # (B, S, F) => Binary Encoding of Missing Values
        x_img: torch.Tensor,  # (B, S, C, H, W)
    ):
        # tabular
        B, S, F = x.shape

        # tabular + time series serialization
        x_flat = self.time_series_serialization(x)
        x_bemv_flat = self.time_series_serialization(x_bemv)

        # 범주형 변수 포함 변환
        x_flat = self.embedder(x_flat)

        # tabular Missing Pattern Independence Encoder (MPIE)
        step_outputs, M_loss = self.mpie(x_flat, x_bemv_flat)
        tab_tok = torch.sum(torch.stack(step_outputs, dim=0), dim=0)
        tab_tok = self.time_series_grouping(tab_tok, x.shape)

        print("tab_tok before view", tab_tok.shape)
        exit()




        # image
        B2, S2, C, H, W = x_img.shape
        x_img_flat = x_img.view(B2 * S2, C, H, W)          # (B*S, C, H, W)
        img_feat = self.image_encoder(x_img_flat)          # (B*S, C, H, W) 그대로
        img_tok = img_feat.view(B2, S2, -1)                # (B, S, n_img = C*H*W)
        print("img_tok 차원", img_tok.shape)

        # ----- 융합(컨캣→프로젝션) -----
        z = self.fusion(tab_tok, img_tok)                  # (B, S, d_model)
        print("융합 후 z 차원", z.shape)

        # ----- 포지셔널 임베딩 + Transformer -----
        z = self.pos_emb(z)                                # (B, S, d_model)
        z = self.transformer(z)                            # (B, S, d_model)

        # ----- 시퀀스 풀링 + 멀티태스크 분류 -----
        h = self.seq_pool(z)                               # (B, d_model)
        outs = self.classifier(h)                          # List[(B, Ck)]

        print("outs.shape", [o.shape for o in outs])
        print("M_loss.shape", M_loss.shape if isinstance(M_loss, torch.Tensor) else M_loss)
        print("z.shape", z.shape)
        exit()

        # outs: 멀티태스크 로짓 리스트, M_loss: TabNet sparsity 정규화 항,
        # z: Transformer의 토큰 임베딩 (필요 시 사용)
        return outs, M_loss, z






if __name__ == "__main__":
    B = 8
    S = 30
    F = 8
    IMG_C = 3
    IMG_H = 112
    IMG_W = 112

    model = MyModel(
        input_dim=8,
        output_dim=[4, 4],
        image_input_size=IMG_H,
    )

    model.to("cuda")
    tabular_dummy = torch.rand((B, S, F), dtype=torch.float32).to("cuda")
    image_dummy = torch.rand((B, S, IMG_C, IMG_H, IMG_W), dtype=torch.float32).to("cuda")
    out = model.forward(tabular_dummy, tabular_dummy, image_dummy)


