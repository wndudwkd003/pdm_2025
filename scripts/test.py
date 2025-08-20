# main.py
from src.utils.seeds import set_seeds
from configs.config import Config
from src.data.dataset_manager import DatasetManager
from src.data.collator_manager import CollatorManager
from src.models.trainer_manager import TrainerManager
from pathlib import Path
import json

cfg = Config()
set_seeds(cfg.seed)

LOAD_MODEL = Path(cfg.load_model)

def prepare_test_dataset(data_dir: str, test_dolder: str):
    test_data_dir = Path(data_dir) / test_dolder
    jsonl_files = list(test_data_dir.glob("*.jsonl"))
    return jsonl_files



def save_result(results: dict, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():

    test_files = prepare_test_dataset(cfg.data_dir, cfg.test)
    dataset_cls = DatasetManager.get_class(cfg.data_type)

    test_ds = dataset_cls(jsonl_files=test_files)
    print(f"Test dataset size: {len(test_ds)}")

    trainer = TrainerManager.get_trainer(
        cfg.model_type,
        work_dir=LOAD_MODEL,
        data_collator=CollatorManager.get_collator(cfg.data_type, cfg.model_type),
    )

    trainer.load(LOAD_MODEL / "final")

    results = trainer.eval(test_dataset=test_ds)

    save_result(results, LOAD_MODEL / "results")


    print(f"Evaluation completed. Results saved to {LOAD_MODEL / 'results'}")


if __name__ == "__main__":
    main()
