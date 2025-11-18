# src/data/collator_class/collator_custom/mptms/tabnet_collator.py

from src.data.collator_class.collator_base.base_collator import BaseCollator

import numpy as np
import torch
import json
from typing import Any

class MPTMSTabNetCollator(BaseCollator):
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        masking_ratio: float = 0.0,
        masking_mode: str = "none",          # "none" | "mcar" | "block_t" | "per_sensor"
        append_mask_indicator: bool = True,  # 마스크 인디케이터 컬럼 추가
        mask_fill: float = 0.0,              # 마스킹 시 채울 값
        csv_has_header: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            append_mask_indicator=append_mask_indicator,
        )
        self.dtype = dtype
        self.masking_ratio = float(masking_ratio)
        self.masking_mode = masking_mode
        self.append_mask_indicator = append_mask_indicator
        self.mask_fill = float(mask_fill)
        self.csv_has_header = csv_has_header
        self.rng = np.random.default_rng(seed)

    def _flatten_csvs(self, csv_paths: list[str]) -> tuple[np.ndarray, int, int]:
        # 시계열 펼치기: 각 CSV의 행(1행) 벡터를 이어붙임 -> (D,)
        vecs: list[np.ndarray] = []
        skip_header = 1 if self.csv_has_header else 0
        feat_dim = None

        for i, csv_path in enumerate(csv_paths):
            arr = np.genfromtxt(csv_path, delimiter=',', dtype=np.float32, skip_header=skip_header)
            if arr.ndim == 1:      # (N,) -> (1, N)
                arr = arr[None, :]
            row = arr[0]            # 1행만 사용: (F,)
            if feat_dim is None:
                feat_dim = row.shape[0]
            vecs.append(row.astype(np.float32))

        x = np.concatenate(vecs, axis=0)     # (T*F,) = (D,)
        T = len(csv_paths)
        F = int(feat_dim) if feat_dim is not None else 0
        return x, F, T

    def _flatten_jsons(self, json_paths: list[str]) -> np.ndarray:
        ys: list[int] = []
        for p in json_paths:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            state = d["annotations"][0]["tagging"][0]["state"]   # "0" | "1" | "2" | "3"
            ys.append(int(state))
        return np.asarray(ys, dtype=np.int64)  # (T_out,)

    def _apply_mask(self, x: np.ndarray, F: int, T: int) -> tuple[np.ndarray, np.ndarray]:
        """
        x: (D,) = (T*F,)
        반환: (x_masked: (D,), obs_mask: (D,))
        obs_mask: 관측 = 1.0, 마스킹 = 0.0
        """
        D = x.shape[0]
        obs = np.ones(D, dtype=np.float32)

        # ------------------------
        # 마스킹 없음
        # ------------------------
        if self.masking_mode == "none" or self.masking_ratio <= 0.0:
            return x, obs

        r = self.masking_ratio

        # =====================================================
        # 정확한 비율을 보장하는 MCAR (권장)
        # =====================================================
        if self.masking_mode == "mcar":
            # 정확히 round(r * D) 개만큼 마스킹
            k = int(round(r * D))
            if k > 0:
                # 마스킹될 index D개 중 k개 선택
                masked_idx = self.rng.choice(D, size=k, replace=False)

                x_masked = x.copy()
                x_masked[masked_idx] = self.mask_fill
                obs[masked_idx] = 0.0

                return x_masked, obs

            # k == 0이면 마스킹 없음
            return x, obs

        # =====================================================
        # block_t: 타임스텝 단위 전체 피처 마스킹
        # =====================================================
        if self.masking_mode == "block_t":
            k = int(round(r * T))
            if k > 0:
                ts = self.rng.choice(T, size=min(k, T), replace=False)
                x_masked = x.copy()
                for t in ts:
                    s = t * F
                    e = s + F
                    obs[s:e] = 0.0
                    x_masked[s:e] = self.mask_fill
                return x_masked, obs
            return x, obs

        # =====================================================
        # per_sensor: 특정 피처(센서) 전체 시계열 마스킹
        # =====================================================
        if self.masking_mode == "per_sensor":
            s_cnt = int(round(r * F))
            if s_cnt > 0:
                fs = self.rng.choice(F, size=min(s_cnt, F), replace=False)
                x_masked = x.copy()
                for t in range(T):
                    base = t * F
                    for f in fs:
                        idx = base + int(f)
                        obs[idx] = 0.0
                        x_masked[idx] = self.mask_fill
                return x_masked, obs
            return x, obs

        # 기본 반환
        return x, obs


    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        sample_ids: list[str] = []
        metas: list[dict[str, Any]] = []

        for sample in batch:
            sid = sample["sample_id"]
            meta = sample["metadata"]

            sample_ids.append(sid)
            metas.append(meta)

            x_vec, F, T = self._flatten_csvs(sample["input_files"]["csvs"])
            x_masked, obs_mask = self._apply_mask(x_vec, F=F, T=T)

            if self.append_mask_indicator:
                x_out = np.concatenate([x_masked, obs_mask], axis=0)  # (D + D,)
            else:
                x_out = x_masked

            Xs.append(x_out)
            y_vec = self._flatten_jsons(sample["target_files"]["labels"])
            ys.append(y_vec)

        X = torch.from_numpy(np.stack(Xs, axis=0)).to(self.dtype)     # (B, D) 또는 (B, 2D)
        y = torch.from_numpy(np.stack(ys, axis=0))                    # (B, T_out), int64로 유지

        return {
            "x": X,
            "y": y,
            "sample_ids": sample_ids,
            "metadata": metas,
        }
