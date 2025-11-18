import torch
import torch.nn as nn

from src.models.core.module.tabular_encoder import (
    FeatTransformer,
    AttentiveTransformer
)

from src.models.core.module.dmattention import (
    MaskedSelfAttention
)

# Missing Pattern Independent Encoder (MPIE)

class MPIEncoder(nn.Module):

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
            input_dim=self.input_dim,
            output_dim=self.feat_output_dim,
            shared_layers=self.shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )


        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()
        self.msa_transformers = nn.ModuleList()

        for _ in range(self.n_steps):
            feat_transformer = FeatTransformer(
                input_dim=self.input_dim,
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

            msa = MaskedSelfAttention(
                d_model=1,
                n_heads=1
            )

            self.feat_transformers.append(feat_transformer)
            self.att_transformers.append(attention)
            self.msa_transformers.append(msa)

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
        # print("after bn x:", x[0, :8])

        feat_out = self.initial_splitter(x)
        att = feat_out[:, self.n_d:]

        # print("x.shape:", x.shape)

        for step in range(self.n_steps):
            # attentive transformer
            tab_mask = self.att_transformers[step](prior, att)

            M_loss += torch.mean(
                torch.sum(tab_mask * torch.log(tab_mask + self.epsilon), dim=1)
            )

            # update prior
            prior = prior * (self.gamma - tab_mask)

            # masked feature
            M_feature_level = tab_mask @ self.group_attention_matrix
            x_masked = M_feature_level * x

            # print("x_masked:", x_masked[0, :8])

            x_masked, A = self.msa_transformers[step](x_masked, bemv)
            attention_maps.append(A)

            # feature transformer
            feat_out = self.feat_transformers[step](x_masked)

            feature_part = feat_out[:, :self.n_d]
            activated = self.activate(feature_part)
            step_outputs.append(activated)

            # update attention
            att = feat_out[:, self.n_d:]

        M_loss /= self.n_steps

        return step_outputs, M_loss, attention_maps










