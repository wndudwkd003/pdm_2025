import torch
import torch.nn as nn

from src.models.core.module.normalizations import GBN
from src.models.core.utils.weights import initialize_glu


class GLU_Block(torch.nn.Module):
    """
    Independent GLU block, specific to each step
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_glu=2,
        first=False,
        shared_layers=None,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()
        self.register_buffer("scale", torch.sqrt(torch.tensor(0.5)))

        for glu_id in range(self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(
                GLU_Layer(
                    input_dim if glu_id == 0 else output_dim,
                    output_dim,
                    fc=fc,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
            ))

    def forward(self, x):
        # the first layer of the block has no scale multiplication
        # 첫 번째 GLU 레이어는 스케일링 없이 바로 통과

        if self.first:
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * self.scale
        return x



class GLU_Layer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        fc=None,
        virtual_batch_size=128,
        momentum=0.02
    ):
        super().__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(
                input_dim,
                2 * output_dim,
                bias=False
            )
        initialize_glu(
            module=self.fc,
            input_dim=input_dim,
            output_dim=2 * output_dim
        )
        self.bn = GBN(
            input_dim=2 * output_dim,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum
        )
        self.glu = nn.GLU(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.glu(x)  # == x[:, :d] * sigmoid(x[:, d:])
        return x
