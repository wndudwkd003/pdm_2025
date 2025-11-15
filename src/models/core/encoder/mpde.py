import torch
import torch.nn as nn

from src.models.core.module.tabular_encoder import (
    FeatTransformer,
    AttentiveTransformer
)
from src.models.core.module.dmattention import (
    MaskedSelfAttention
)
from src.models.core.module.layer_module import GLU_Block
from src.models.core.utils.weights import initialize_non_glu


class SelfAttention1D(nn.Module):
    """
    특징 축(F)에 대한 간단한 self-attention.
    입력  x: (B, F)  →  출력 y: (B, F), attn: (B, F, F)
    """
    def __init__(self, d_model: int = 1, n_heads: int = 1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        q = k = v = x.unsqueeze(-1)        # (B, F, 1)
        y, attn = self.mha(q, k, v, need_weights=True)
        return y.squeeze(-1), attn         # (B, F), (B, F, F)


# -----------------------------
# GLUBranch (Attention 없음)
# -----------------------------
class GLUBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        virtual_batch_size: int,
        momentum: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

        self.glu_blocks = nn.ModuleList()
        self.fcs = nn.ModuleList()

        for i in range(n_layers):
            self.glu_blocks.append(
                GLU_Block(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
                )
            )
            fc = nn.Linear(input_dim, input_dim, bias=True)
            initialize_non_glu(fc, input_dim, input_dim)
            self.fcs.append(fc)

        self.activate = nn.ReLU()
        self.to_hidden = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        z = x
        for idx, (glu, fc) in enumerate(zip(self.glu_blocks, self.fcs), start=1):
            z = glu(z)
            z = fc(z)


        z = self.activate(z)
        hidden = self.to_hidden(z)

        return hidden


# -----------------------------
# MPDEncoder (Attention → GLUBranch) × steps
# -----------------------------
class MPDEncoder(nn.Module):


    group_attention_matrix: torch.Tensor


    def __init__(
        self,
        input_dim: int,
        n_d: int,
        n_a: int,
        n_shared: int,
        n_independent: int,
        n_steps: int,
        virtual_batch_size: int,
        momentum: float,
        group_attention_matrix: torch.Tensor,
        mask_type: str,
        bias: bool,
        epsilon: float,
        gamma: float,
    ):
        super().__init__()

        self.register_buffer(
            "group_attention_matrix",
            group_attention_matrix.to(torch.float32)
        )

        self.input_dim = input_dim

        self.input_dim_mpde = input_dim * 2 # bemv 추가

        self.n_d = n_d
        self.n_a = n_a
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.n_steps = n_steps
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.attention_dim = self.group_attention_matrix.shape[0]
        self.bias = bias
        self.feat_output_dim = self.n_d + self.n_a
        self.mask_type = mask_type
        self.epsilon = epsilon
        self.gamma = gamma

        self.initial_bn = nn.BatchNorm1d(
            num_features=self.input_dim,
            momentum=self.momentum
        )

        self.shared_feat_transform = nn.ModuleList([
            nn.Linear(
                in_features=self.input_dim if i == 0 else self.feat_output_dim,
                out_features=2 * (self.feat_output_dim),
                bias=self.bias
            )
            for i in range(self.n_shared)
        ])

        self.initial_splitter = FeatTransformer(
            input_dim=self.input_dim_mpde,
            output_dim=self.feat_output_dim,
            shared_layers=self.shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )


        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()
        self.sa_transformers = nn.ModuleList()

        for _ in range(self.n_steps):
            feat_transformer = FeatTransformer(
                input_dim=self.input_dim_mpde,
                output_dim=self.feat_output_dim,
                shared_layers=self.shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )

            attention = AttentiveTransformer(
                input_dim=self.n_a,
                group_dim=self.attention_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                mask_type=self.mask_type,
            )

            sa = SelfAttention1D(
                d_model=1,
                n_heads=1
            )

            self.feat_transformers.append(feat_transformer)
            self.att_transformers.append(attention)
            self.sa_transformers.append(sa)

        self.activate = nn.LeakyReLU() # nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,        # input feature
        bemv: torch.Tensor,     # binary encoding of missing values

    ):
        # init prior
        prior = torch.ones((x.shape[0], self.attention_dim), device=x.device)

        # init outputs
        M_loss = 0.0
        step_outputs = []
        attention_maps = []

        # init batch norm
        x = self.initial_bn(x)

        x_cat = torch.cat([x, bemv], dim=1) # 결측 의존 특징 추가

        feat_out = self.initial_splitter(x_cat)
        att = feat_out[:, self.n_d:]


        for step in range(self.n_steps):
            # attentive transformer
            tab_mask = self.att_transformers[step](prior, att)

            # print("tab_mask:", tab_mask[0, :8])
            # print("tab_mask.shape", tab_mask.shape)

            M_loss += torch.mean(torch.sum(tab_mask * torch.log(tab_mask + self.epsilon), dim=1))

            # update prior
            prior = prior * (self.gamma - tab_mask)

            # masked feature
            M_feature_level = tab_mask @ self.group_attention_matrix
            x_masked = M_feature_level * x

            # print("x_masked:", x_masked[0, :8])

            x_masked, A = self.sa_transformers[step](x_masked, bemv)
            attention_maps.append(A)

            # feature transformer
            x_masked_cat = torch.cat([x_masked, bemv], dim=1)
            feat_out = self.feat_transformers[step](x_masked_cat)

            feature_part = feat_out[:, :self.n_d]
            activated = self.activate(feature_part)
            step_outputs.append(activated)

            # update attention
            att = feat_out[:, self.n_d:]

        M_loss /= self.n_steps

        return step_outputs, M_loss, attention_maps






