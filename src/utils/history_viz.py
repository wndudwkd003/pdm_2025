# src/utils/history_viz.py
from pathlib import Path
import json
import matplotlib.pyplot as plt

def _plot_line(series: list[float], xlab: str, ylab: str, title: str, outpath: Path):
    plt.figure()
    plt.plot(series)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_history_graphs(history: dict, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if "loss" in history and isinstance(history["loss"], list) and len(history["loss"]) > 0:
        _plot_line(history["loss"], "Epoch", "loss", "Train Loss", save_dir / "train_loss.png")

    if "lr" in history and isinstance(history["lr"], list) and len(history["lr"]) > 0:
        _plot_line(history["lr"], "Epoch", "lr", "Learning Rate", save_dir / "lr.png")

    keys = list(history.keys())
    train_metrics = [k for k in keys if k.startswith("train_")]
    for tk in train_metrics:
        mk = tk[len("train_"):]
        vk = f"valid_{mk}"
        if vk in history and isinstance(history[tk], list) and isinstance(history[vk], list):
            plt.figure()
            plt.plot(history[tk], label=tk)
            plt.plot(history[vk], label=vk)
            plt.xlabel("Epoch")
            plt.ylabel(mk)
            plt.title(mk)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_dir / f"{mk}.png", dpi=150)
            plt.close()
        else:
            if isinstance(history[tk], list) and len(history[tk]) > 0:
                _plot_line(history[tk], "Epoch", tk, tk, save_dir / f"{tk}.png")

def save_history_artifacts(history: dict, save_dir: Path):
    """history.json 저장 + 그래프 생성"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    save_history_graphs(history, save_dir)

def plot_history_from_model_dir(model_dir: Path):
    """{model_dir}/history/history.json을 읽어 그래프만 다시 생성"""
    hist_path = Path(model_dir) / "history" / "history.json"
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    save_history_graphs(history, hist_path.parent)
