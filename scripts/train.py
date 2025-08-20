# main.py
from src.utils.seeds import set_seeds
from configs.config import Config
from src.data.dataset_manager import DatasetManager
from src.data.collator_manager import CollatorManager
from src.models.trainer_manager import TrainerManager
from src.utils.path_util import get_save_path
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.history_viz import save_history_artifacts


cfg = Config()
set_seeds(cfg.seed)


def prepare_split_dataset(data_dir: str, train_folder: str, seed: int, split_ratio: float):
    train_data_dir = Path(data_dir) / train_folder
    jsonl_files = list(train_data_dir.glob("*.jsonl"))
    train_files, valid_files = train_test_split(jsonl_files, test_size=split_ratio, random_state=seed)
    print(f"Found total {len(jsonl_files)} files.")
    return train_files, valid_files


def main():
    save_dir = get_save_path(cfg.model_type, cfg.data_type, cfg.save_dir)

    train_files, valid_files = prepare_split_dataset(cfg.data_dir, cfg.train, cfg.seed, cfg.split_ratio)
    dataset_cls = DatasetManager.get_class(cfg.data_type)

    train_ds = dataset_cls(jsonl_files=train_files)
    valid_ds = dataset_cls(jsonl_files=valid_files)
    print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(valid_ds)}")

    trainer = TrainerManager.get_trainer(
        cfg.model_type,
        work_dir=save_dir,
        data_collator=CollatorManager.get_collator(cfg.data_type, cfg.model_type),
    )

    history = trainer.fit(train_dataset=train_ds, valid_dataset=valid_ds)

    save_history_artifacts(history, save_dir / "history")

    save_path = trainer.save(save_dir / "final")

    print(f"Training completed and model saved to {save_dir}")


if __name__ == "__main__":
    main()
