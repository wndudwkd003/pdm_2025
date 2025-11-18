# main.py
import os
import torch
import torch.distributed as dist
torch.set_float32_matmul_precision("high")  # 또는 "medium"
from src.utils.seeds import set_seeds
import os




from configs.config import Config
from src.data.dataset_manager import DatasetManager
from src.data.collator_manager import CollatorManager
from src.models.trainer_manager import TrainerManager
from src.utils.path_util import get_save_path
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.history_viz import save_history_artifacts
from argparse import ArgumentParser
from src.configs.config_manager import ConfigManager
from src.utils.feature_name import expand_feature_names


def prepare_split_dataset(data_dir: str, train_folder: str, seed: int, split_ratio: float):
    train_data_dir = Path(data_dir) / train_folder
    jsonl_files = list(train_data_dir.glob("*.jsonl"))
    train_files, valid_files = train_test_split(jsonl_files, test_size=split_ratio, random_state=seed)
    print(f"Found total {len(jsonl_files)} files.")
    return train_files, valid_files


def main(cfg: Config):
    save_dir = get_save_path(
        cfg.model_type, cfg.data_type, cfg.save_dir,
        cfg.masking_ratio, cfg.masking_mode
    )

    # -------------------------
    # DDP 초기화
    # -------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    is_main_rank = (local_rank == 0)

    train_files, valid_files = prepare_split_dataset(cfg.data_dir, cfg.train, cfg.seed, cfg.split_ratio)
    dataset_cls = DatasetManager.get_class(cfg.data_type)

    train_ds = dataset_cls(jsonl_files=train_files)
    valid_ds = dataset_cls(jsonl_files=valid_files)

    metadata = train_ds.meta

    if is_main_rank:
        print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(valid_ds)}")

    model_config = ConfigManager.get_model_config(cfg.model_type)
    if use_ddp:
        model_config.device = f"cuda:{local_rank}"
    else:
        model_config.device = "cuda"

    trainer = TrainerManager.get_trainer(
        model_type=cfg.model_type,
        work_dir=save_dir,
        data_collator=CollatorManager.get_collator(cfg.data_type, cfg.model_type)(
            masking_ratio=cfg.masking_ratio,
            masking_mode=cfg.masking_mode,
            append_mask_indicator=cfg.append_mask_indicator,
            mask_fill=cfg.mask_fill,
            csv_has_header=cfg.csv_has_header,
            seed=cfg.seed,
        ),
        model_config=model_config,
        metadata=metadata
    )

    history = trainer.fit(train_dataset=train_ds, valid_dataset=valid_ds)

    if is_main_rank:
        save_history_artifacts(history, save_dir / "history")
        save_path = trainer.save(save_dir / "final")
        print(f"Training completed and model saved to {save_path.absolute()}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    cfg = Config()
    set_seeds(cfg.seed)
    main(cfg)
