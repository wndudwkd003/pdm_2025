import torch
import torch.nn as nn

from src.models.core.module.tabular_encoder import (
    FeatTransformer,
    AttentiveTransformer
)
from src.models.core.module.dmattention import (
    SelfAttention1D
)
from src.models.core.module.layer_module import GLU_Block
from src.models.core.utils.weights import initialize_non_glu


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
                in_features=self.input_dim_mpde if i == 0 else self.feat_output_dim,
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

            self.feat_transformers.append(
                FeatTransformer(
                    input_dim=self.input_dim_mpde,
                    output_dim=self.feat_output_dim,
                    shared_layers=self.shared_feat_transform,
                    n_glu_independent=self.n_independent,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=self.momentum,
                )
            )

            self.att_transformers.append(
                AttentiveTransformer(
                    input_dim=self.n_a,
                    group_dim=self.attention_dim,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=self.momentum,
                    mask_type=self.mask_type,
                )
            )

            self.sa_transformers.append(
                SelfAttention1D(
                    d_model=1,
                    n_heads=1
                )
            )

        self.activate = nn.LeakyReLU() # nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,        # input feature
        bemv: torch.Tensor,     # binary encoding of missing values

    ):
        B, F = x.shape

        # init prior (TabNet-style)
        prior = torch.ones((B, self.attention_dim), device=x.device)

        M_loss = 0.0
        step_outputs: list[torch.Tensor] = []
        attention_maps: list[torch.Tensor] = []

        # x 에만 BN
        x = self.initial_bn(x)                    # (B, F)

        # bemv 와 concat
        x_cat = torch.cat([x, bemv], dim=1)       # (B, 2F)



        feat_out = self.initial_splitter(x_cat)   # (B, n_d + n_a)


        att = feat_out[:, self.n_d:]              # (B, n_a)

        for step in range(self.n_steps):
            # feature selection mask
            tab_mask = self.att_transformers[step](prior, att)   # (B, G)

            # sparsity loss
            M_loss += torch.mean(
                torch.sum(tab_mask * torch.log(tab_mask + self.epsilon), dim=1)
            )

            # prior 업데이트
            prior = prior * (self.gamma - tab_mask)

            # group-level mask → feature-level mask
            M_feature_level = tab_mask @ self.group_attention_matrix   # (B, F)
            x_masked = M_feature_level * x                             # (B, F)

            # Self-Attention (missing 의존 정보는 x_masked, bemv concat 에 이미 반영)
            x_attn, A = self.sa_transformers[step](x_masked)           # (B, F), (B, F, F)
            attention_maps.append(A)

            # 다시 bemv 붙여서 feature transformer 통과
            x_attn_cat = torch.cat([x_attn, bemv], dim=1)              # (B, 2F)
            feat_out = self.feat_transformers[step](x_attn_cat)        # (B, n_d + n_a)

            feature_part = feat_out[:, :self.n_d]                      # (B, n_d)
            activated = self.activate(feature_part)
            step_outputs.append(activated)

            # 다음 step용 attention 입력
            att = feat_out[:, self.n_d:]

        M_loss /= self.n_steps



        return step_outputs, M_loss, attention_maps



class MPDDecoder(nn.Module):
    """
    Missing-Pattern Dependent Decoder (MPDD)

    - 입력:  h_agg (B, n_d)      # MPDEncoder 가 내주는 aggregated hidden
    - 출력:  x_rec (B, input_dim)  # MPDEncoder 의 입력 차원 (post_embed_dim) 으로 복원
    """
    def __init__(
        self,
        input_dim: int,          # 복원할 차원 (예: post_embed_dim)
        n_d: int,
        n_layers: int,
        virtual_batch_size: int,
        momentum: float,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_layers = n_layers
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum

        self.glu_blocks = nn.ModuleList()
        self.fcs = nn.ModuleList()

        for _ in range(n_layers):
            self.glu_blocks.append(
                GLU_Block(
                    input_dim=self.n_d,
                    output_dim=self.n_d,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=self.momentum,
                )
            )
            fc = nn.Linear(self.n_d, self.n_d, bias=True)
            initialize_non_glu(fc, self.n_d, self.n_d)
            self.fcs.append(fc)

        self.activate = nn.ReLU()
        self.to_output = nn.Linear(self.n_d, self.input_dim)

    def forward(self, h_agg: torch.Tensor):
        # h_agg: (B, n_d)
        z = h_agg
        for glu, fc in zip(self.glu_blocks, self.fcs):
            z = glu(z)      # (B, n_d)
            z = fc(z)       # (B, n_d)

        z = self.activate(z)
        x_rec = self.to_output(z)     # (B, input_dim)

        return x_rec
