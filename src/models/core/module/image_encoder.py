import torch

HUB_REPO = "facebookresearch/dinov2"

class DINOv2EncoderHub(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = torch.hub.load(HUB_REPO, model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model(x)
        return feat
