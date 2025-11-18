# scripts/test.py

from src.utils.seeds import set_seeds
from configs.config import Config
from src.data.dataset_manager import DatasetManager
from src.data.collator_manager import CollatorManager
from src.models.trainer_manager import TrainerManager
from pathlib import Path
from src.utils.eval_viz import save_eval_artifacts
from src.utils.history_viz import plot_history_from_model_dir

from argparse import ArgumentParser
from xgboost import XGBModel

from src.configs.config_manager import ConfigManager

LOAD_MODEL = None

def prepare_test_dataset(data_dir: str, test_dolder: str):
    test_data_dir = Path(data_dir) / test_dolder
    jsonl_files = list(test_data_dir.glob("*.jsonl"))
    return jsonl_files

def main(cfg: Config):
    test_files = prepare_test_dataset(cfg.data_dir, cfg.test)
    dataset_cls = DatasetManager.get_class(cfg.data_type)

    test_ds = dataset_cls(jsonl_files=test_files)
    print(f"Test dataset size: {len(test_ds)}")

    # 메타데이터 설정
    metadata = test_ds.meta
    model_config = ConfigManager.get_model_config(cfg.model_type) # instance of ModelConfig


    trainer = TrainerManager.get_trainer(
        model_type=cfg.model_type,
        work_dir=LOAD_MODEL,
        data_collator=CollatorManager.get_collator(cfg.data_type, cfg.model_type)(
            masking_ratio=cfg.masking_ratio,
            masking_mode=cfg.masking_mode,
            append_mask_indicator=cfg.append_mask_indicator,
            mask_fill=cfg.mask_fill,
            csv_has_header=cfg.csv_has_header,
            seed=cfg.seed,
        ),
        # 모델 설정은 ConfigManager를 통해 가져옴
        model_config=model_config,
        metadata=metadata
    )

    trainer.load(LOAD_MODEL / "final")


    results = trainer.eval(test_dataset=test_ds) # , tde=tde)

    save_eval_artifacts(results, LOAD_MODEL / "results")

    # 학습때 저장된 history.json이 있다면, 여기서 그래프만 다시 생성
    plot_history_from_model_dir(LOAD_MODEL)

    print(f"Evaluation completed. Results saved to {(LOAD_MODEL / 'results').absolute()}")

if __name__ == "__main__":
    cfg = Config()

    set_seeds(cfg.seed)

    LOAD_MODEL = Path(cfg.load_model)

    main(cfg)
