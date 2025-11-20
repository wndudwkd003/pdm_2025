from sklearn.model_selection import train_test_split

from pathlib import Path


def prepare_split_dataset(data_dir: str, train_folder: str, seed: int, split_ratio: float):
    train_data_dir = Path(data_dir) / train_folder
    jsonl_files = list(train_data_dir.glob("*.jsonl"))
    train_files, valid_files = train_test_split(jsonl_files, test_size=split_ratio, random_state=seed)
    print(f"Found total {len(jsonl_files)} files.")
    return train_files, valid_files
