# scripts/train.py
import torch
import torch.distributed as dist
torch.set_float32_matmul_precision("high")

import os
from pathlib import Path   # 추가

from configs.config import Config
from src.data.dataset_manager import DatasetManager as dm
from src.data.collator_manager import CollatorManager as cm
from src.models.trainer_manager import TrainerManager as tm
from src.configs.config_manager import ConfigManager as cfgm

from src.utils.seeds import set_seeds
from src.utils.history_viz import save_history_artifacts
from src.utils.path_util import get_save_path
from src.utils.split_data import prepare_split_dataset


def main(cfg: Config):
    save_dir = get_save_path(
        cfg.model_type, cfg.data_type, cfg.save_dir,
        cfg.masking_ratio, cfg.masking_mode
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    is_main_rank = (local_rank == 0)

    train_files, valid_files = prepare_split_dataset(
        cfg.data_dir, cfg.train, cfg.seed, cfg.split_ratio
    )

    dt_cls = dm.get_class(cfg.data_type)
    train_ds = dt_cls(jsonl_files=train_files)
    valid_ds = dt_cls(jsonl_files=valid_files)

    if is_main_rank:
        print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(valid_ds)}")

    model_config = cfgm.get_model_config(cfg.model_type)

    if use_ddp:
        model_config.device = f"cuda:{local_rank}"


    trainer = tm.get_trainer(
        model_type=cfg.model_type,
        work_dir=save_dir,
        data_collator=cm.get_collator(cfg.data_type, cfg.model_type)(
            masking_ratio=cfg.masking_ratio,
            masking_mode=cfg.masking_mode,
            append_mask_indicator=cfg.append_mask_indicator,
            mask_fill=cfg.mask_fill,
            csv_has_header=cfg.csv_has_header,
            seed=cfg.seed,
            multimodal_setting=model_config.multimodal_setting,
            masking_scinario_augmentation=cfg.masking_scinario_augmentation,
            masking_ratio_pro=cfg.masking_ratio_pro,
        ),
        model_config=model_config,
        metadata=train_ds.meta
    )

    # ★ finetune 모드일 때만 사전학습 weight 로드
    if model_config.training_mode == "finetune":
        preload_dir = Path(cfg.load_model) / "final"
        print(f"[Finetune] Load pretrained weights from: {preload_dir}")
        trainer.load(preload_dir)

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
