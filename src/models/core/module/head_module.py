import torch
import torch.nn as nn
from typing import Sequence as sequ
from src.models.core.utils.weights import initialize_non_glu

class Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: sequ[int],
    ):
        """
        멀티태스크 분류 헤드.
        output_dim: 각 태스크의 클래스 개수 리스트 (예: [3, 5, 2])
        항상 List[Tensor] 형태로 반환합니다: [ (B, C1), (B, C2), ... ].
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.multi_task_mappings = nn.ModuleList()
        for task_dim in output_dim:
            head = nn.Linear(input_dim, task_dim, bias=True)
            initialize_non_glu(head, input_dim, task_dim)
            self.multi_task_mappings.append(head)

    def forward(self, x: torch.Tensor):
        # x: (B, input_dim)
        outs = []
        for head in self.multi_task_mappings:
            outs.append(head(x))  # (B, task_dim)
        return outs



