from pathlib import Path

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer

class TabNetTrainer(BaseTrainer):
    def __init__(
        self,
        work_dir: Path,
        data_collator: BaseCollator,
    ):
        super().__init__(
            work_dir=work_dir,
            data_collator=data_collator,
        )


    def fit(
        self,
        train_dataset,
        valid_dataset,
    ):
        pass


    def save(
        self,
        save_path: Path,
    ):
        return save_path


    def load(
        self,
        model_path: Path,
    ):
        pass


