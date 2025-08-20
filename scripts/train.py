# main.py
from src.utils.seeds import set_seeds
from configs.config import Config
from src.data.dataset_manager import DatasetManager
from src.data.collator_manager import CollatorManager
from src.models.trainer_manager import TrainerManager
from src.utils.path_util import get_save_path

from pathlib import Path
from sklearn.model_selection import train_test_split

cfg = Config()
set_seeds(cfg.seed)


def prepare_split_dataset(data_dir: str, train_folder: str, seed: int, split_ratio: float):
    train_data_dir = Path(data_dir) / train_folder
    jsonl_files = list(train_data_dir.glob("*.jsonl"))
    train_files, valid_files = train_test_split(jsonl_files, test_size=split_ratio, random_state=seed)
    print(f"Found total {len(jsonl_files)} files.")
    return train_files, valid_files

# main.py 등 동일 파일에 두거나, util 모듈로 분리하셔도 됩니다.
import matplotlib.pyplot as plt
from pathlib import Path

def save_history(history, save_path: Path):
    """
    TabNet history 딕셔너리로부터 그래프를 저장합니다.
    - loss           -> train_loss.png
    - lr             -> lr.png
    - train_* / valid_* -> 각 키 이름 그대로 {key}.png
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    def _plot(series, title, filename, ylabel=None):
        plt.figure()
        plt.plot(series)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel if ylabel is not None else title)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path / filename, dpi=150)
        plt.close()

    # 1) 학습 손실 곡선
    if "loss" in history and isinstance(history["loss"], list) and len(history["loss"]) > 0:
        _plot(history["loss"], "Train Loss", "train_loss.png", ylabel="loss")

    # 2) 학습률 곡선(있을 때만)
    if "lr" in history and isinstance(history["lr"], list) and len(history["lr"]) > 0:
        _plot(history["lr"], "Learning Rate", "lr.png", ylabel="lr")

    # 3) train_*, valid_* 지표들 자동 저장
    for k, v in history.items():
        if not isinstance(v, list) or len(v) == 0:
            continue
        if k.startswith("train_") or k.startswith("valid_"):
            _plot(v, k, f"{k}.png", ylabel=k)


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

    history = trainer.fit(
        train_dataset=train_ds,
        valid_dataset=valid_ds,
    )

    save_history(history, save_dir / "history")

    save_path = trainer.save(save_dir / "final")

    print(f"Training completed and model saved to {save_dir}")


if __name__ == "__main__":
    main()
