import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config import Config
from src.data.dataset_manager import DatasetManager as dm
from src.data.collator_manager import CollatorManager as cm
from src.configs.config_manager import ConfigManager as cfgm

from src.models.core.HDBE import HybridDoubleBranchEncoder
from src.utils.seeds import set_seeds
from src.utils.split_data import prepare_split_dataset


def build_dataloader(dataset, cfg: Config):
    model_config = cfgm.get_model_config(cfg.model_type)

    collator_cls = cm.get_collator(cfg.data_type, cfg.model_type)
    collator = collator_cls(
        masking_ratio=cfg.masking_ratio,
        masking_mode=cfg.masking_mode,
        append_mask_indicator=cfg.append_mask_indicator,
        mask_fill=cfg.mask_fill,
        csv_has_header=cfg.csv_has_header,
        seed=cfg.seed,
        multimodal_setting=False,
        masking_scinario_augmentation=cfg.masking_scinario_augmentation,
        masking_ratio_pro=cfg.masking_ratio_pro,
    )
    loader = DataLoader(
        dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    return loader


def convert_zero_nan(x: torch.Tensor, bemv: torch.Tensor):
    missing_mask = bemv == 0
    x = x.clone()
    x[missing_mask] = 0.0
    return x


def extract_embeddings(
    model: HybridDoubleBranchEncoder,
    loader,
    device: torch.device,
    desc: str,
):
    model.eval()
    all_latent = []
    all_y = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=desc):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            bemv = (~torch.isnan(x)).to(x.dtype)
            x_zero = convert_zero_nan(x, bemv)

            out = model(x_zero, bemv)
            latent = out["latent"]

            all_latent.append(latent.cpu())
            all_y.append(y.cpu())
            all_ids.extend(batch["sample_ids"])

    X = torch.cat(all_latent, dim=0).numpy()
    Y = torch.cat(all_y, dim=0).numpy()
    ids = np.array(all_ids)

    return X, Y, ids


def prepare_test_dataset(data_dir: str, test_folder: str):
    test_data_dir = Path(data_dir) / test_folder
    jsonl_files = list(test_data_dir.glob("*.jsonl"))
    return jsonl_files


def main(cfg: Config):
    train_files, valid_files = prepare_split_dataset(
        cfg.data_dir, cfg.train, cfg.seed, cfg.split_ratio
    )
    test_files = prepare_test_dataset(cfg.data_dir, cfg.test)

    dt_cls = dm.get_class(cfg.data_type)
    train_ds = dt_cls(jsonl_files=train_files)
    valid_ds = dt_cls(jsonl_files=valid_files)
    test_ds = dt_cls(jsonl_files=test_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = train_ds.feature_dim
    embed_dim = 64

    model = HybridDoubleBranchEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
    ).to(device)

    state = torch.load(cfg.load_model + "/final/save_model", map_location=device)
    model.load_state_dict(state)

    train_loader = build_dataloader(train_ds, cfg)
    valid_loader = build_dataloader(valid_ds, cfg)
    test_loader = build_dataloader(test_ds, cfg)

    X_tr, y_tr, ids_tr = extract_embeddings(model, train_loader, device, desc="train")
    X_va, y_va, ids_va = extract_embeddings(model, valid_loader, device, desc="valid")
    X_te, y_te, ids_te = extract_embeddings(model, test_loader, device, desc="test")

    out_dir = Path(cfg.save_dir) / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "r02"

    np.savez(out_dir / f"train_hdbe_{prefix}.npz", X=X_tr, y=y_tr, ids=ids_tr)
    np.savez(out_dir / f"valid_hdbe_{prefix}.npz", X=X_va, y=y_va, ids=ids_va)
    np.savez(out_dir / f"test_hdbe_{prefix}.npz", X=X_te, y=y_te, ids=ids_te)

if __name__ == "__main__":
    cfg = Config()
    set_seeds(cfg.seed)
    main(cfg)
