# src/data/dataset_base/base_dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import List, Dict, Any

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.samples: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def load_data(self):
        raise NotImplementedError
