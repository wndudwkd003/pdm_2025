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
    all_concat = []   # latent || original_flat
    all_add = []      # latent (+) original_flat (왼쪽부터 더하기)
    all_y = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=desc):
            x = batch["x"].to(device)  # (B, S, F)
            y = batch["y"].to(device)

            # BEMV 및 결측 → 0 처리
            bemv = (~torch.isnan(x)).to(x.dtype)
            x_zero = convert_zero_nan(x, bemv)

            # 원본 시계열을 B, S*F 로 펼친 벡터
            B = x_zero.size(0)
            original_flat = x_zero.view(B, -1)  # (B, S*F)

            # 모델 forward
            out = model(x_zero, bemv)
            latent = out["latent"]  # (B, D_latent) 가정

            # 1) 임베딩과 원본 벡터 concat
            concat_feat = torch.cat([latent, original_flat], dim=1)  # (B, D_latent + S*F)

            # 2) 임베딩 + 원본값을 왼쪽에서 더하는 방식
            d_latent = latent.size(1)
            d_orig = original_flat.size(1)

            if d_latent >= d_orig:
                # 긴 쪽이 latent
                add_feat = latent.clone()
                add_feat[:, :d_orig] += original_flat
            else:
                # 긴 쪽이 original_flat
                add_feat = original_flat.clone()
                add_feat[:, :d_latent] += latent

            all_concat.append(concat_feat.cpu())
            all_add.append(add_feat.cpu())
            all_y.append(y.cpu())
            all_ids.extend(batch["sample_ids"])

    X_concat = torch.cat(all_concat, dim=0).numpy()
    X_add = torch.cat(all_add, dim=0).numpy()
    Y = torch.cat(all_y, dim=0).numpy()
    ids = np.array(all_ids)

    return X_concat, X_add, Y, ids


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

    device = cfgm.get_model_config(cfg.model_type).device

    input_dim = train_ds.feature_dim
    embed_dim = 64

    model = HybridDoubleBranchEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
    ).to(device)
    model.eval()

    state = torch.load(cfg.load_model + "/final/save_model", map_location=device)
    model.load_state_dict(state)

    train_loader = build_dataloader(train_ds, cfg)
    valid_loader = build_dataloader(valid_ds, cfg)
    test_loader = build_dataloader(test_ds, cfg)

    # 임베딩 + 원본(flat) concat, 그리고 왼쪽 더하기 버전 둘 다 추출
    X_tr_concat, X_tr_add, y_tr, ids_tr = extract_embeddings(
        model, train_loader, device, desc="train"
    )
    X_va_concat, X_va_add, y_va, ids_va = extract_embeddings(
        model, valid_loader, device, desc="valid"
    )
    X_te_concat, X_te_add, y_te, ids_te = extract_embeddings(
        model, test_loader, device, desc="test"
    )

    out_dir = Path(cfg.save_dir) / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "r04"

    # 기존 파일 이름은 "임베딩 + 원본 concat" 결과로 유지
    np.savez(out_dir / f"train_hdbe_{prefix}.npz", X=X_tr_concat, y=y_tr, ids=ids_tr)
    np.savez(out_dir / f"valid_hdbe_{prefix}.npz", X=X_va_concat, y=y_va, ids=ids_va)
    np.savez(out_dir / f"test_hdbe_{prefix}.npz", X=X_te_concat, y=y_te, ids=ids_te)

    # 임베딩 + 원본값을 왼쪽부터 더한 벡터도 추가로 저장
    np.savez(out_dir / f"train_hdbe_{prefix}_add.npz", X=X_tr_add, y=y_tr, ids=ids_tr)
    np.savez(out_dir / f"valid_hdbe_{prefix}_add.npz", X=X_va_add, y=y_va, ids=ids_va)
    np.savez(out_dir / f"test_hdbe_{prefix}_add.npz", X=X_te_add, y=y_te, ids=ids_te)


if __name__ == "__main__":
    cfg = Config()
    set_seeds(cfg.seed)
    main(cfg)
