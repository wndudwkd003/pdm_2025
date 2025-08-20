# src/data/mptms_dataset.py
from pathlib import Path
import json
from typing import List, Dict, Any
from src.data.dataset_class.dataset_base.base_dataset import BaseDataset

class MPTMSDataset(BaseDataset):
    def __init__(self, jsonl_files: List[Path]=None):
        super().__init__()

        self.jsonl_files = [Path(p) for p in jsonl_files]
        self.load_data()

    def load_data(self):
        samples: List[Dict[str, Any]] = []
        for fpath in self.jsonl_files:
            with open(fpath, "r", encoding="utf-8") as f:
                objs = [json.loads(line.strip()) for line in f]
                samples.extend(objs)

        self.samples = samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]
