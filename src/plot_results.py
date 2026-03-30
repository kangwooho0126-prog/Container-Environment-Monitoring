import math
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURE_DIR, PRED_RESULT_CSV
from utils import ensure_dir


def plot_multi_container(df, true_col, pred_col, title, ylabel, save_path):
    container_ids = sorted(df["container_number"].unique())
    n = len(container_ids)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False)
    axes = axes.flatten()

    for ax, container_id in zip(axes, container_ids):
        sub = df[df["container_number"] == container_id].copy()

        if "prediction_time" in sub.columns:
            sub["prediction_time"] = pd.to_datetime(sub["prediction_time"])
            sub = sub.sort_values("prediction_time")
            x = sub["prediction_time"]
        else:
            x = range(len(sub))

        ax.plot(x, sub[true_col], label="True", linewidth=2)
        ax.plot(x, sub[pred_col], label="Pred", linewidth=2)
        ax.set_title(str(container_id))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    eval_dir = FIGURE_DIR / "evaluation"
    ensure_dir(eval_dir)

    df = pd.read_csv(PRED_RESULT_CSV)

    required = {
        "container_number",
        "temperature_true",
        "temperature_pred",
        "humidity_true",
        "humidity_pred",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in prediction file: {sorted(missing)}")

    plot_multi_container(
        df,
        "temperature_true",
        "temperature_pred",
        "Temperature Prediction vs True by Container",
        "Temperature",
        eval_dir / "temperature_by_container.png",
    )

    plot_multi_container(
        df,
        "humidity_true",
        "humidity_pred",
        "Humidity Prediction vs True by Container",
        "Humidity",
        eval_dir / "humidity_by_container.png",
    )

    print(f"Saved evaluation figures to: {eval_dir}")


if __name__ == "__main__":
    main()