# src/models/core/module/attention_viz.py

from pathlib import Path
from typing import List, Literal, Callable

import torch
import matplotlib.pyplot as plt


def log_attention(
    *,
    epoch: int,
    phase: str,
    attn_log_dir: Path,
    attn_type: Literal["mpie", "transformer"],
    x_dbg: torch.Tensor,                 # (B, S, F_orig)
    bemv_dbg: torch.Tensor,              # (B, S, F_orig)
    attention_maps_dbg: List[torch.Tensor],  # 각 step의 attn 텐서 리스트
    time_series_serialization_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> None:
    """
    MPIE / Transformer 공통 Attention 시각화 함수.

    Parameters
    ----------
    attn_type : "mpie" | "transformer"
        - "mpie"        : feature-level attention (F_attn x F_attn)
        - "transformer" : time-step attention (S x S)
    x_dbg : (B, S, F_orig)
        원본 입력 시퀀스 (shape 정보용)
    bemv_dbg : (B, S, F_orig)
        결측 여부 (1/0)
    attention_maps_dbg : list of Tensor
        각 step의 attention map 텐서
        - MPIE        : (B, H, F_attn, F_attn)
        - Transformer : (B, H, S, S)
    time_series_serialization_fn :
        (B, S, F) -> (B*S, F) 로 flatten 해주는 함수 (MyModel.time_series_serialization)
    """

    if len(attention_maps_dbg) == 0:
        print(
            f"[AttentionViz] Empty {attn_type} attention_maps "
            f"(phase={phase}, epoch={epoch})"
        )
        return

    x_shape = x_dbg.shape
    B_dbg = x_shape[0]
    S_dbg = x_shape[1]
    F_orig = x_shape[2]

    # 공통: 첫 번째 step, 첫 번째 sample, 첫 번째 head만 사용
    step_idx = 0
    A_step = attention_maps_dbg[step_idx]  # (B, H, D1, D2)
    A_shape = A_step.shape
    N_dbg = A_shape[0]
    H_dbg = A_shape[1]
    D1 = A_shape[2]
    D2 = A_shape[3]

    sample_idx = 0
    head_idx = 0

    A_sample = A_step[sample_idx, head_idx]  # (D1, D2)

    # -------------------------------------------------
    # 1) MPIE (feature-level) attention
    # -------------------------------------------------
    if attn_type == "mpie":
        # bemv flatten: (B, S, F_orig) -> (B*S, F_orig)
        x_bemv_flat_dbg = time_series_serialization_fn(
            bemv_dbg.to(device)
        )
        bemv_sample = x_bemv_flat_dbg[sample_idx]  # (F_flat,)

        F_attn = D1
        F_bemv = bemv_sample.shape[0]
        F_min = min(F_attn, F_orig, F_bemv)

        A_cpu = A_sample.detach().cpu()
        bemv_cpu = bemv_sample.detach().cpu()

        # --- 텍스트 로그 ---
        txt_path = attn_log_dir / f"mpie_attn_epoch_{epoch:04d}_{phase}.txt"
        with open(txt_path, "w") as f:
            f.write(f"Epoch {epoch} / phase={phase}\n")
            f.write(f"sample_idx={sample_idx}, head_idx={head_idx}\n")
            f.write("bemv (1=observed, 0=missing):\n")
            f.write("  " + str(bemv_cpu.tolist()) + "\n\n")

            for f_idx in range(F_min):
                if bemv_cpu[f_idx].item() == 0.0:
                    row_sum = A_cpu[f_idx, :].sum().item()
                    col_sum = A_cpu[:, f_idx].sum().item()
                    f.write(
                        f"feature {f_idx} missing: row_sum={row_sum:.6e}, "
                        f"col_sum={col_sum:.6e}\n"
                    )

        # --- 히트맵 PNG ---
        png_path = attn_log_dir / f"mpie_attn_epoch_{epoch:04d}_{phase}.png"

        plt.figure(figsize=(5, 4))
        plt.imshow(A_cpu, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title(
            f"MPIE Epoch {epoch} / {phase} / "
            f"sample {sample_idx} / head {head_idx}"
        )

        missing_indices = []
        for f_idx in range(F_min):
            if bemv_cpu[f_idx].item() == 0.0:
                missing_indices.append(f_idx)

        for f_idx in missing_indices:
            plt.axhline(y=f_idx - 0.5, linewidth=0.5)
            plt.axhline(y=f_idx + 0.5, linewidth=0.5)
            plt.axvline(x=f_idx - 0.5, linewidth=0.5)
            plt.axvline(x=f_idx + 0.5, linewidth=0.5)

        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        return

    # -------------------------------------------------
    # 2) Transformer (time-step) attention
    # -------------------------------------------------
    if attn_type == "transformer":
        A_cpu = A_sample.detach().cpu()

        png_path = attn_log_dir / f"tr_attn_epoch_{epoch:04d}_{phase}.png"

        plt.figure(figsize=(5, 4))
        plt.imshow(A_cpu, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title(
            f"Transformer Epoch {epoch} / {phase} / "
            f"sample {sample_idx} / head {head_idx}"
        )
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        return
