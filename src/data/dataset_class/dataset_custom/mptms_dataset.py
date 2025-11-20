# src/data/mptms_dataset.py
from pathlib import Path
import json
from typing import Any
from src.data.dataset_class.dataset_base.base_dataset import BaseDataset

class MPTMSDataset(BaseDataset):
    def __init__(self, jsonl_files: list[Path]=None):
        super().__init__()
        self.jsonl_files = [Path(p) for p in jsonl_files]
        self.meta = {}
        self.feature_dim = None
        self.seq_length = None
        self.load_data()

    def load_data(self):
        samples = []
        for fpath in self.jsonl_files:
            with open(fpath, "r", encoding="utf-8") as f:
                objs = [json.loads(line.strip()) for line in f]
                samples.extend(objs)

        self.samples = samples
        self.meta = self.samples[0]["metadata"]


        continuous_cols = self.meta["continuous_cols"]
        categorical_cols = self.meta["categorical_cols"]

        self.feature_dim = len(continuous_cols) + len(categorical_cols)
        self.seq_length = self.meta["forward"]

    def __getitem__(self, index: int):
        return self.samples[index]
