from pathlib import Path
import json
import math
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


def _get_series(history: dict, key: str):
    """
    history[key]가 '비어 있지 않은 list'인 경우에만 반환하고,
    아니면 None을 반환하는 공통 유틸 함수.
    """
    if key in history:
        series = history[key]
        if isinstance(series, list) and len(series) > 0:
            return series
    return None


def _calc_percentile(values: list[float], q: float) -> float | None:
    """
    values에서 q(0~1) 퍼센타일 값을 계산.
    values가 비어 있으면 None 반환.
    """
    if len(values) == 0:
        return None

    sorted_vals = sorted(values)
    # 0 <= q <= 1 가정
    idx = int((len(sorted_vals) - 1) * q)
    return sorted_vals[idx]

def _plot_metric_with_zoom(
    series_list: list[list[float]],
    labels: list[str],
    xlab: str,
    ylab: str,
    title: str,
    save_dir: Path,
    basename: str,
    zoom_percentile: float = 0.95,
):
    plt.figure()
    for s, lab in zip(series_list, labels):
        plt.plot(s, label=lab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{basename}.png", dpi=150)
    plt.close()

    all_values: list[float] = []
    for s in series_list:
        for v in s:
            if isinstance(v, (int, float)):
                fv = float(v)
                if math.isfinite(fv):
                    all_values.append(fv)

    if len(all_values) == 0:
        return

    low = min(all_values)
    high = _calc_percentile(all_values, zoom_percentile)
    if high is None:
        return

    if not (high > low):
        return

    span = high - low
    margin = span * 0.05
    y_min = low - margin
    y_max = high + margin

    plt.figure()
    for s, lab in zip(series_list, labels):
        plt.plot(s, label=lab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(f"{title} (zoom)")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(save_dir / f"{basename}_zoom.png", dpi=150)
    plt.close()



def save_history_graphs(history: dict, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1) loss: train_loss / valid_loss 한 그림에 같이
    #    + zoom 버전(loss_zoom.png) 추가
    # -------------------------------------------------
    train_loss = _get_series(history, "train_loss")
    valid_loss = _get_series(history, "valid_loss")

    if train_loss is not None and valid_loss is not None:
        # 원본 + zoom 두 개 저장
        _plot_metric_with_zoom(
            series_list=[train_loss, valid_loss],
            labels=["train_loss", "valid_loss"],
            xlab="Epoch",
            ylab="loss",
            title="loss",
            save_dir=save_dir,
            basename="loss",          # loss.png, loss_zoom.png
            zoom_percentile=0.95,     # 상위 5%는 잘라내는 기준
        )
    else:
        # 혹시 옛날 형식(history["loss"])을 쓸 수도 있으니 남겨둠
        loss_series = _get_series(history, "loss")
        if loss_series is not None:
            _plot_line(loss_series, "Epoch", "loss", "Train Loss", save_dir / "train_loss.png")

    # -------------------------------------------------
    # 2) Learning Rate 곡선 (옵션)
    # -------------------------------------------------
    lr_series = _get_series(history, "lr")
    if lr_series is not None:
        _plot_line(lr_series, "Epoch", "lr", "Learning Rate", save_dir / "lr.png")

    # -------------------------------------------------
    # 3) 그 외 train_* / valid_* 페어 공통 처리
    #    예: train_acc / valid_acc → acc.png
    # -------------------------------------------------
    keys = list(history.keys())
    train_metrics = [k for k in keys if k.startswith("train_")]

    for tk in train_metrics:
        mk = tk[len("train_"):]  # 예: "train_acc" → "acc"

        # 위에서 loss는 이미 처리했으니 여기선 건너뜀
        if mk == "loss":
            pass
        else:
            vk = f"valid_{mk}"
            train_series = _get_series(history, tk)
            valid_series = _get_series(history, vk)

            if train_series is not None and valid_series is not None:
                plt.figure()
                plt.plot(train_series, label=tk)
                plt.plot(valid_series, label=vk)
                plt.xlabel("Epoch")
                plt.ylabel(mk)
                plt.title(mk)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(save_dir / f"{mk}.png", dpi=150)
                plt.close()
            else:
                # valid_*가 없으면 train_*만 단독으로라도 그림
                if train_series is not None:
                    _plot_line(train_series, "Epoch", tk, tk, save_dir / f"{tk}.png")


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
