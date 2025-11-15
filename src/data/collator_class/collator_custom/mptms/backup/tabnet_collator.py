# src/data/collator_class/collator_custom/mptms/tabnet_collator.py

from src.data.collator_class.collator_base.base_collator import BaseCollator

import numpy as np
import torch
import json
from typing import Any

class MPTMSTabNetCollator(BaseCollator):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dtype = dtype

    def _flatten_csvs(self, csv_paths: list[str]) -> np.ndarray:
        vecs: list[np.ndarray] = []

        for csv_path in csv_paths:
            arr = np.genfromtxt(
                csv_path,
                delimiter=',',
                dtype=np.float32,
                skip_header=1,
            )
            if arr.ndim == 1: # # 1행 CSV 대비 (N,) -> (1, N)
                arr = arr[None, :]
            vecs.append(arr.ravel())

        return np.concatenate(vecs, axis=0) # (N, D) -> (D,) 1D vector


    def _flatten_jsons(self, json_paths: list[str]) -> np.ndarray:
        ys: list[int] = []
        for p in json_paths:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            state = d["annotations"][0]["tagging"][0]["state"]  # "0"|"1"|"2"|"3"
            ys.append(int(state))
        return np.asarray(ys, dtype=np.int64)  # (T_out,)


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

            x_vec = self._flatten_csvs(sample["input_files"]["csvs"])
            Xs.append(x_vec)

            y_vec = self._flatten_jsons(sample["target_files"]["labels"])
            ys.append(y_vec)


        X = torch.from_numpy(np.stack(Xs, axis=0)).to(self.dtype) # (B, D)
        y = torch.from_numpy(np.stack(ys, axis=0)).to(self.dtype) # (B, T_out)

        return {
            "x": X,
            "y": y,
            "sample_ids": sample_ids,
            "metadata": metas,
        }
