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
        self.load_data()

    def load_data(self):
        samples: list[dict[str, Any]] = []
        for fpath in self.jsonl_files:
            with open(fpath, "r", encoding="utf-8") as f:
                objs = [json.loads(line.strip()) for line in f]
                samples.extend(objs)

        self.samples = samples
        self.meta = self.samples[0]["metadata"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
