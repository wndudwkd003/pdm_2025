from pathlib import Path
import json
import matplotlib
import matplotlib.pyplot as plt

# 필요하면 주석 해제해서 non-GUI 환경에서도 동작하게 할 수 있습니다.
# matplotlib.use("Agg")


def _plot_line(x_values, y_values, xlab, ylab, title, outpath):
    plt.figure()
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def load_metrics_by_missing_rate_range(start_dir: Path, end_dir: Path, run_prefix: str):
    """
    start_dir ~ end_dir 사이(이름 기준) 폴더들 중에서
    이름에 run_prefix 가 포함된 것만 골라 metrics.json 을 읽습니다.

    예)
      start_dir = outputs/2025-11-17_15-44-12_xgboost_mptms_none_0.0
      end_dir   = outputs/2025-11-17_15-48-48_xgboost_mptms_mcar_0.8

      -> 이 둘이 있는 상위 폴더(outputs) 안에서
         이름이 start.name ~ end.name 사이인 xgboost_mptms 실험만 사용
    """
    start_dir = Path(start_dir)
    end_dir = Path(end_dir)

    if start_dir.parent != end_dir.parent:
        raise ValueError("start_dir과 end_dir는 같은 상위 디렉터리 아래에 있어야 합니다.")

    root = start_dir.parent

    start_name = start_dir.name
    end_name = end_dir.name

    # start / end 순서가 바뀌어 들어와도 처리
    if start_name > end_name:
        tmp = start_name
        start_name = end_name
        end_name = tmp

    mr_to_metrics = {}

    # 상위 폴더 안의 실험 디렉터리들 순회
    for run_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if run_dir.is_dir():
            name = run_dir.name

            # 이름 범위 필터
            if name < start_name:
                pass
            elif name > end_name:
                pass
            else:
                # prefix 필터
                if run_prefix in name:
                    parts = name.split("_")
                    mr_str = parts[-1]
                    missing_rate = float(mr_str)

                    metrics_path = run_dir / "results" / "metrics.json"
                    if metrics_path.is_file():
                        with metrics_path.open("r", encoding="utf-8") as f:
                            metrics = json.load(f)
                        mr_to_metrics[missing_rate] = metrics

    missing_rates = sorted(mr_to_metrics.keys())
    metrics_list = [mr_to_metrics[mr] for mr in missing_rates]
    return missing_rates, metrics_list


def plot_overall_metrics_vs_missing_rate(missing_rates, metrics_list, save_dir: Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    accuracy_list = []
    precision_macro_list = []
    recall_macro_list = []
    f1_macro_list = []

    for m in metrics_list:
        overall = m["overall"]
        accuracy_list.append(overall["accuracy"])
        precision_macro_list.append(overall["precision_macro"])
        recall_macro_list.append(overall["recall_macro"])
        f1_macro_list.append(overall["f1_macro"])

    # Accuracy
    _plot_line(
        missing_rates,
        accuracy_list,
        "Missing rate (MCAR)",
        "Accuracy",
        "Accuracy vs Missing rate",
        save_dir / "overall_accuracy_vs_missing_rate.png",
    )

    # Precision (macro)
    _plot_line(
        missing_rates,
        precision_macro_list,
        "Missing rate (MCAR)",
        "Precision (macro)",
        "Macro Precision vs Missing rate",
        save_dir / "overall_precision_macro_vs_missing_rate.png",
    )

    # Recall (macro)
    _plot_line(
        missing_rates,
        recall_macro_list,
        "Missing rate (MCAR)",
        "Recall (macro)",
        "Macro Recall vs Missing rate",
        save_dir / "overall_recall_macro_vs_missing_rate.png",
    )

    # F1 (macro)
    _plot_line(
        missing_rates,
        f1_macro_list,
        "Missing rate (MCAR)",
        "F1 (macro)",
        "Macro F1 vs Missing rate",
        save_dir / "overall_f1_macro_vs_missing_rate.png",
    )


if __name__ == "__main__":
    # 예시: 질문에서 주신 start / end 경로
    start_dir = Path(
        "outputs/2025-11-18_15-10-39_xgboost_c-mapss_none_0.0"
    )
    end_dir = Path(
        "outputs/2025-11-18_16-59-58_xgboost_c-mapss_mcar_0.8"
    )

    run_prefix = "xgboost_c-mapss"

    # 요약 그래프 저장 위치
    save_dir = start_dir.parent / "summary_plots_xgboost_c-mapss_range"

    missing_rates, metrics_list = load_metrics_by_missing_rate_range(
        start_dir=start_dir,
        end_dir=end_dir,
        run_prefix=run_prefix,
    )

    plot_overall_metrics_vs_missing_rate(missing_rates, metrics_list, save_dir)
