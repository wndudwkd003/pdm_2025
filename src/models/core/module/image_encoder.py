import torch
import torch.nn as nn

HUB_REPO = "facebookresearch/dinov2"


class DINOv2EncoderHub(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool = True):
        super().__init__()
        self.model = torch.hub.load(HUB_REPO, model_name)
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            # 초기 상태를 eval로 고정
            self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            # gradient / activation 메모리 아끼기
            with torch.no_grad():
                feat = self.model(x)
        else:
            feat = self.model(x)
        return feat

    def train(self, mode: bool = True):
        # MyModel.train() 호출 시에도 backbone은 계속 eval로 유지
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self
