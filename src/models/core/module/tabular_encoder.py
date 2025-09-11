import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet import sparsemax, tab_model
from src.models.core.module.normalizations import GBN
from src.models.core.utils.weights import initialize_non_glu
from src.models.core.module.layer_module import GLU_Block



class AttentiveTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        group_dim,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        group_dim : int
            Number of groups for features
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, group_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, group_dim)
        self.bn = GBN(
            group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = sparsemax.Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError(
                "Please choose either sparsemax" + "or entmax as masktype"
            )

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x





class FeatTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        shared_layers,
        n_glu_independent,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Output_size
        shared_layers : torch.nn.ModuleList
            The shared block that should be common to every step
        n_glu_independent : int
            Number of independent GLU layers
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization within GLU block(s)
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(
                spec_input_dim,
                output_dim,
                first=is_first,
                n_glu=n_glu_independent,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum
            )

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x





class TabNetEncoder(nn.Module):
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
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.n_steps = n_steps
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.register_buffer(
            "group_attention_matrix",
            group_attention_matrix.to(torch.float32)
        )
        self.attention_dim = self.group_attention_matrix.shape[0]
        self.bias = bias
        self.feat_output_dim = self.n_d + self.n_a
        self.mask_type = mask_type
        self.epsilon = epsilon
        self.gamma = gamma

        self.initial_bn = nn.BatchNorm1d(
            num_features=self.input_dim,
            momentum=0.01
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

            self.feat_transformers.append(feat_transformer)
            self.att_transformers.append(attention)

        self.activate = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        prior: torch.Tensor=None
    ):

        print("bn 전 ------")
        print(x)
        print("----")
        x = self.initial_bn(x)


        print("bn 후 차원:", x.shape)
        print("bn 후 ------")
        print(x)
        print("----")

        B = x.shape[0] # batch size

        if prior is None:
            prior = torch.ones((B, self.attention_dim)).to(x.device)

        print("prior 차원 및 값 ")
        print(prior.shape)
        print(prior)

        feat_out = self.initial_splitter(x)
        print("init feat out 차원 : ", feat_out.shape)

        att = feat_out[:, self.n_d:] # (B, n_a)

        print("att 차원 및 값 ")
        print(att.shape)
        print(att)

        M_loss = 0.0
        step_outputs = []

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            print("M Attentive transformer 출력 차원")
            print(M.shape)
            print(M)
            print("---")

            print("---")
            print("self.group_attention_matrix shape: ", self.group_attention_matrix.shape)
            print(self.group_attention_matrix)
            print("---")

            M_loss += torch.mean(torch.sum(M * torch.log(M + self.epsilon), dim=1))

            # update prior
            prior = (self.gamma - M) * prior

            # output
            M_feature_level = M @ self.group_attention_matrix

            # 마스킹 적용
            masked_x = M_feature_level * x

            # 마스킹 적용된 입력을 feat transformer에 통과
            feat_out = self.feat_transformers[step](masked_x)

            print("feat_out shape: ", feat_out.shape)
            print(feat_out)

            # feature 부분 n_d 까지만
            feature_part = feat_out[:, :self.n_d]
            activated = self.activate(feature_part)
            step_outputs.append(activated)

            # 다음 att 업데이트
            att = feat_out[:, self.n_d:]

        M_loss = M_loss / self.n_steps
        return step_outputs, M_loss





