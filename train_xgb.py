import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from configs.config import Config
from src.utils.seeds import set_seeds
from src.configs.config_manager import ConfigManager
from src.utils.metrics import compute_multitask_classification_metrics
from src.utils.eval_viz import save_eval_artifacts


def load_embeddings(npz_path: Path):
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    ids = None
    if "ids" in data.files:
        ids = data["ids"]
    return X, y, ids


def train_xgb_from_embeddings(X_tr, y_tr, X_va, y_va, model_config):
    T = y_tr.shape[1]
    models: list[XGBClassifier] = []
    history: dict[str, list[float] | int] = {}

    for t in range(T):
        metric_name = model_config.eval_metric

        clf = XGBClassifier(
            objective=model_config.objective,
            num_class=model_config.num_classes,
            random_state=model_config.seed,
            eval_metric=metric_name,
            early_stopping_rounds=model_config.early_stopping_rounds,
        )

        clf.fit(
            X_tr,
            y_tr[:, t],
            eval_set=[(X_tr, y_tr[:, t]), (X_va, y_va[:, t])],
        )

        models.append(clf)

        evals = clf.evals_result()
        history[f"task{t}_train_{metric_name}"] = evals["validation_0"][metric_name]
        history[f"task{t}_valid_{metric_name}"] = evals["validation_1"][metric_name]
        history[f"task{t}_best_iteration"] = int(clf.best_iteration)

    return models, history


def eval_xgb_on_embeddings(models: list[XGBClassifier], X, y_true):
    preds = []
    for t in range(y_true.shape[1]):
        p = models[t].predict(X)
        preds.append(p)
    y_pred = np.stack(preds, axis=1).astype(np.int64)
    metrics = compute_multitask_classification_metrics(y_true, y_pred)
    return metrics


def save_xgb_models(models: list[XGBClassifier], save_path: Path, model_config):
    save_path.mkdir(parents=True, exist_ok=True)
    meta = {"num_tasks": len(models)}
    with open(save_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    for t, clf in enumerate(models):
        out_name = f"{model_config.save_model_name}{t}.{model_config.model_ext}"
        clf.save_model(str(save_path / out_name))
    return save_path


def main(cfg: Config):
    emb_dir = Path(cfg.save_dir) / "embeddings"

    train_npz = emb_dir / "train_hdbe_r06.npz"
    valid_npz = emb_dir / "valid_hdbe_r06.npz"
    test_npz = emb_dir / "test_hdbe_r06.npz"

    X_tr, y_tr, ids_tr = load_embeddings(train_npz)
    X_va, y_va, ids_va = load_embeddings(valid_npz)

    has_test = test_npz.exists()
    X_te = None
    y_te = None
    ids_te = None
    if has_test:
        X_te, y_te, ids_te = load_embeddings(test_npz)

    model_config = ConfigManager.get_model_config(cfg.model_type)

    work_dir = Path(cfg.save_dir) / "xgb_hdbe_r06"
    work_dir.mkdir(parents=True, exist_ok=True)

    models, history = train_xgb_from_embeddings(X_tr, y_tr, X_va, y_va, model_config)

    history_path = work_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    final_dir = work_dir / "final"
    save_xgb_models(models, final_dir, model_config)

    valid_metrics = eval_xgb_on_embeddings(models, X_va, y_va)
    save_eval_artifacts(valid_metrics, work_dir / "results_valid")

    if has_test:
        test_metrics = eval_xgb_on_embeddings(models, X_te, y_te)
        save_eval_artifacts(test_metrics, work_dir / "results_test")


if __name__ == "__main__":
    cfg = Config()
    set_seeds(cfg.seed)
    main(cfg)
