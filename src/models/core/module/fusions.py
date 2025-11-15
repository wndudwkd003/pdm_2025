import numpy as np
import torch
import torch.nn as nn

class FusionConcat(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.LazyLinear(d_model)   # (n_tab+n_img) -> d_model
        self._ln_tab: nn.LayerNorm = None
        self._ln_img: nn.LayerNorm = None

    def forward(self, tab_tok: torch.Tensor, img_tok: torch.Tensor) -> torch.Tensor:
        # tab_tok: (B, S, n_tab), img_tok: (B, S, n_img)
        if self._ln_tab is None:
            self._ln_tab = nn.LayerNorm(tab_tok.size(-1)).to(tab_tok.device)
        elif next(self._ln_tab.parameters(), torch.empty(0, device=tab_tok.device)).device != tab_tok.device:
            # 장치가 다르면 맞춰줌 (재생성 없이 이동)
            self._ln_tab = self._ln_tab.to(tab_tok.device)

        if self._ln_img is None:
            self._ln_img = nn.LayerNorm(img_tok.size(-1)).to(img_tok.device)
        elif next(self._ln_img.parameters(), torch.empty(0, device=img_tok.device)).device != img_tok.device:
            self._ln_img = self._ln_img.to(img_tok.device)

        t = self._ln_tab(tab_tok)
        i = self._ln_img(img_tok)
        f = torch.cat([t, i], dim=-1)  # (B, S, n_tab+n_img)

        # LazyLinear가 처음 materialize될 때 입력 장치를 따르도록 보장
        if hasattr(self.proj, "weight") and self.proj.weight is not None:
            # 이미 materialize된 경우: 장치 불일치가 있다면 이동
            if self.proj.weight.device != f.device:
                self.proj = self.proj.to(f.device)

        z = self.proj(f)               # (B, S, d_model)
        return z
