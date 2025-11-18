# src/data/collator_class/collator_custom/mptms/mymodel_collator.py
# from src.data.collator_class.collator_base.base_collator import BaseCollator

# import numpy as np
# import torch
# import json
# from typing import Any

# class MPTMSMyModelCollator(BaseCollator):
#     """
#     MyModel 전용 Collator.
#     - x: (B, S, F)  (S = forward, F = feature dim)
#     - 결측/마스킹 위치는 NaN 으로 표시
#     - y: (B, T_out)
#     - BEMV(x_bemv)는 모델 내부에서 torch.isnan(x) 로 생성
#     """
#     def __init__(
#         self,
#         dtype: torch.dtype = torch.float32,
#         masking_ratio: float = 0.0,
#         masking_mode: str = "none",          # "none" | "mcar" | "block_t" | "per_sensor"
#         append_mask_indicator: bool = False, # MyModel 에서는 사용 안 함
#         mask_fill: float = 0.0,              # 사용 안 함 (NaN 사용)
#         csv_has_header: bool = True,
#         seed: int | None = None,
#     ):
#         super().__init__(
#             append_mask_indicator=append_mask_indicator,
#         )
#         self.dtype = dtype
#         self.masking_ratio = float(masking_ratio)
#         self.masking_mode = masking_mode
#         self.append_mask_indicator = append_mask_indicator
#         self.mask_fill = float(mask_fill)
#         self.csv_has_header = csv_has_header
#         self.rng = np.random.default_rng(seed)

#     def _load_csvs_as_ts(self, csv_paths: list[str]) -> tuple[np.ndarray, int, int]:
#         """
#         각 CSV의 1행을 읽어서 시계열로 쌓음.
#         반환:
#             x_ts: (T, F)
#             F: feature dim
#             T: time steps (= len(csv_paths))
#         """
#         rows: list[np.ndarray] = []
#         skip_header = 1 if self.csv_has_header else 0
#         feat_dim = None

#         for csv_path in csv_paths:
#             arr = np.genfromtxt(csv_path, delimiter=',', dtype=np.float32, skip_header=skip_header)
#             if arr.ndim == 1:      # (N,) -> (1, N)
#                 arr = arr[None, :]
#             row = arr[0]            # 1행만 사용: (F,)
#             if feat_dim is None:
#                 feat_dim = row.shape[0]
#             rows.append(row.astype(np.float32))

#         x_ts = np.stack(rows, axis=0)  # (T, F)
#         T = x_ts.shape[0]
#         F = x_ts.shape[1] if feat_dim is not None else 0
#         return x_ts, F, T

#     def _flatten_jsons(self, json_paths: list[str]) -> np.ndarray:
#         ys: list[int] = []
#         for p in json_paths:
#             with open(p, "r", encoding="utf-8") as f:
#                 d = json.load(f)
#             state = d["annotations"][0]["tagging"][0]["state"]   # "0" | "1" | "2" | "3"
#             ys.append(int(state))
#         return np.asarray(ys, dtype=np.int64)  # (T_out,)

#     def _apply_mask_ts(self, x_ts: np.ndarray) -> np.ndarray:
#         """
#         x_ts: (T, F)
#         반환: x_masked_ts: (T, F)  (마스킹 위치를 NaN 으로 세팅)
#         """
#         T, F = x_ts.shape
#         x_masked = x_ts.copy()

#         if self.masking_mode == "none" or self.masking_ratio <= 0.0:
#             return x_masked

#         r = self.masking_ratio

#         # mcar: 완전 랜덤 마스킹
#         if self.masking_mode == "mcar":
#             m = self.rng.random(size=(T, F)) >= r    # True=관측, False=마스킹
#             x_masked[~m] = np.nan
#             return x_masked

#         # block_t: 타임스텝 단위로 전체 피처를 가림
#         if self.masking_mode == "block_t":
#             k = int(round(r * T))                # 가릴 타임스텝 수
#             if k > 0:
#                 ts = self.rng.choice(T, size=min(k, T), replace=False)
#                 for t in ts:
#                     x_masked[t, :] = np.nan
#             return x_masked

#         # per_sensor: 피처(센서) 단위로 전체 타임에 대해 가림
#         if self.masking_mode == "per_sensor":
#             s_cnt = int(round(r * F))
#             if s_cnt > 0:
#                 fs = self.rng.choice(F, size=min(s_cnt, F), replace=False)
#                 for f in fs:
#                     x_masked[:, f] = np.nan
#             return x_masked

#         # 알 수 없는 모드 -> 원본 그대로 반환
#         return x_masked

#     def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
#         Xs: list[np.ndarray] = []   # 각 샘플 x_ts: (S, F)
#         ys: list[np.ndarray] = []   # 각 샘플 y: (T_out,)
#         sample_ids: list[str] = []
#         metas: list[dict[str, Any]] = []

#         for sample in batch:
#             sid = sample["sample_id"]
#             meta = sample["metadata"]

#             sample_ids.append(sid)
#             metas.append(meta)

#             # --- 입력 CSV -> (S, F) ---
#             x_ts, F, S = self._load_csvs_as_ts(sample["input_files"]["csvs"])  # (S, F)

#             # --- 마스킹 적용 (NaN) ---
#             x_ts = self._apply_mask_ts(x_ts)  # (S, F), 마스킹 위치는 NaN

#             Xs.append(x_ts)

#             # --- 타겟 라벨 ---
#             y_vec = self._flatten_jsons(sample["target_files"]["labels"])  # (T_out,)
#             ys.append(y_vec)

#         # (B, S, F), (B, T_out)
#         X = torch.from_numpy(np.stack(Xs, axis=0)).to(self.dtype)
#         y = torch.from_numpy(np.stack(ys, axis=0))

#         return {
#             "x": X,               # (B, S, F)
#             "y": y,               # (B, T_out)
#             "sample_ids": sample_ids,
#             "metadata": metas,
#         }


# src/data/collator_class/collator_custom/mptms/mymodel_collator.py
from src.data.collator_class.collator_base.base_collator import BaseCollator

import numpy as np
import torch
import json
from typing import Any
import cv2


class MPTMSMyModelCollator(BaseCollator):
    """
    MyModel 전용 Collator.
    - x: (B, S, F)  (S = forward, F = feature dim)
    - 결측/마스킹 위치는 NaN 으로 표시
    - y: (B, T_out)
    - BEMV(x_bemv)는 모델 내부에서 torch.isnan(x) 로 생성
    - x_img: (B, S, 3, H, W)  (IR bin → 컬러맵 → 리사이즈)
    """
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        masking_ratio: float = 0.0,
        masking_mode: str = "none",          # "none" | "mcar" | "block_t" | "per_sensor"
        append_mask_indicator: bool = False,
        mask_fill: float = 0.0,
        csv_has_header: bool = True,
        seed: int | None = None,
        image_size: int = 224,
        multimodal_setting: bool = False,
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
        self.image_size = int(image_size)
        self.multimodal_setting = bool(multimodal_setting)

    def _load_csvs_as_ts(self, csv_paths: list[str]) -> tuple[np.ndarray, int, int]:
        rows: list[np.ndarray] = []
        skip_header = 1 if self.csv_has_header else 0
        feat_dim = None

        for csv_path in csv_paths:
            arr = np.genfromtxt(csv_path, delimiter=',', dtype=np.float32, skip_header=skip_header)
            if arr.ndim == 1:
                arr = arr[None, :]
            row = arr[0]
            if feat_dim is None:
                feat_dim = row.shape[0]
            rows.append(row.astype(np.float32))

        x_ts = np.stack(rows, axis=0)  # (T, F)
        T = x_ts.shape[0]
        F = x_ts.shape[1] if feat_dim is not None else 0
        return x_ts, F, T

    def _flatten_jsons(self, json_paths: list[str]) -> np.ndarray:
        ys: list[int] = []
        for p in json_paths:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            state = d["annotations"][0]["tagging"][0]["state"]
            ys.append(int(state))
        return np.asarray(ys, dtype=np.int64)

    def _apply_mask_ts(self, x_ts: np.ndarray) -> np.ndarray:
        T, F = x_ts.shape
        x_masked = x_ts.copy()

        if self.masking_mode == "none" or self.masking_ratio <= 0.0:
            return x_masked

        r = self.masking_ratio

        if self.masking_mode == "mcar":
            k = int(round(r * F))
            if k <= 0:
                return x_masked
            if k >= F:
                k = F - 1

            for t in range(T):
                cols = self.rng.choice(F, size=k, replace=False)
                x_masked[t, cols] = np.nan

            return x_masked

        if self.masking_mode == "block_t":
            k = int(round(r * T))
            if k > 0:
                ts = self.rng.choice(T, size=min(k, T), replace=False)
                for t in ts:
                    x_masked[t, :] = np.nan
            return x_masked

        if self.masking_mode == "per_sensor":
            s_cnt = int(round(r * F))
            if s_cnt > 0:
                fs = self.rng.choice(F, size=min(s_cnt, F), replace=False)
                for f in fs:
                    x_masked[:, f] = np.nan
            return x_masked

        return x_masked

    def _load_images_as_ts(self, image_paths: list[str]) -> np.ndarray:
        """
        image_paths: 각 타임스텝마다 1개의 .bin 파일 경로 리스트
        반환: imgs_ts: (T, 3, H, W)
        """
        imgs: list[np.ndarray] = []
        H = self.image_size
        W = self.image_size

        for bin_path in image_paths:
            ir_image = np.load(bin_path)                  # (h, w), IR 값
            ir_norm = cv2.normalize(
                ir_image,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            img_colored = cv2.applyColorMap(ir_norm, cv2.COLORMAP_JET)  # (h, w, 3)
            img_resized = cv2.resize(img_colored, (W, H))
            img = img_resized.astype(np.float32) / 255.0                # (H, W, 3)
            img = np.transpose(img, (2, 0, 1))                          # (3, H, W)
            imgs.append(img)

        imgs_ts = np.stack(imgs, axis=0)  # (T, 3, H, W)
        return imgs_ts

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        Xs_original: list[np.ndarray] = []
        Xs: list[np.ndarray] = []
        X_imgs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        sample_ids: list[str] = []
        metas: list[dict[str, Any]] = []

        for sample in batch:
            sid = sample["sample_id"]
            meta = sample["metadata"]

            sample_ids.append(sid)
            metas.append(meta)

            x_ts, F, S = self._load_csvs_as_ts(sample["input_files"]["csvs"])

            x_ts_original = x_ts.copy()
            x_ts_masked = self._apply_mask_ts(x_ts)

            Xs_original.append(x_ts_original)
            Xs.append(x_ts_masked)

            # 이미지 시퀀스 로딩
            if self.multimodal_setting:
                image_paths = sample["input_files"].get("images", None)
                if image_paths is not None:
                    img_ts = self._load_images_as_ts(image_paths)  # (S_img, 3, H, W)
                    if img_ts.shape[0] != S:
                        raise ValueError(
                            f"CSV 시퀀스 길이(S={S})와 이미지 시퀀스 길이({img_ts.shape[0]})가 다릅니다."
                        )
                    X_imgs.append(img_ts)

            y_vec = self._flatten_jsons(sample["target_files"]["labels"])
            ys.append(y_vec)

        X_original = torch.from_numpy(np.stack(Xs_original, axis=0)).to(self.dtype)
        X = torch.from_numpy(np.stack(Xs, axis=0)).to(self.dtype)
        y = torch.from_numpy(np.stack(ys, axis=0))

        batch_dict = {
            "x": X,
            "x_original": X_original,
            "y": y,
            "sample_ids": sample_ids,
            "metadata": metas,
        }

        if self.multimodal_setting and len(X_imgs) > 0:
            X_img = torch.from_numpy(np.stack(X_imgs, axis=0))  # (B, S, 3, H, W)
            batch_dict["x_img"] = X_img

        return batch_dict
