import re
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURE_DIR, PRED_RESULT_CSV
from utils import ensure_dir


def classify_risk_level(risk_prob):
    if risk_prob >= 0.7:
        return "Critical"
    if risk_prob >= 0.4:
        return "Warning"
    return "Normal"


def compute_run_lengths(mask):
    max_run = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _safe_name(text):
    return re.sub(r'[\\/*?:"<>|()]', "", str(text))


def run_alert_pipeline(
    product_name,
    temp_warn_low,
    temp_warn_high,
    temp_crit_low,
    temp_crit_high,
    hum_warn_low,
    hum_warn_high,
    hum_crit_low,
    hum_crit_high,
    horizon=72,
):
    df = pd.read_csv(PRED_RESULT_CSV)

    required_cols = {
        "container_number",
        "prediction_time",
        "temperature_pred",
        "humidity_pred",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    df = df.sort_values(["container_number", "prediction_time"]).copy()

    figure_subdir = FIGURE_DIR / product_name
    ensure_dir(figure_subdir)

    output_rows = []
    grouped_data = []

    for container_id, group in df.groupby("container_number"):
        group = group.sort_values("prediction_time").tail(horizon).copy()

        temp_warn_mask = (
            (group["temperature_pred"] < temp_warn_low)
            | (group["temperature_pred"] > temp_warn_high)
        )
        temp_crit_mask = (
            (group["temperature_pred"] < temp_crit_low)
            | (group["temperature_pred"] > temp_crit_high)
        )

        hum_warn_mask = (
            (group["humidity_pred"] < hum_warn_low)
            | (group["humidity_pred"] > hum_warn_high)
        )
        hum_crit_mask = (
            (group["humidity_pred"] < hum_crit_low)
            | (group["humidity_pred"] > hum_crit_high)
        )

        warn_mask = temp_warn_mask | hum_warn_mask
        crit_mask = temp_crit_mask | hum_crit_mask

        warn_hours = int(warn_mask.sum())
        crit_hours = int(crit_mask.sum())

        longest_warn_run = compute_run_lengths(warn_mask.tolist())
        longest_crit_run = compute_run_lengths(crit_mask.tolist())

        total_points = max(len(group), 1)
        risk_prob = min(
            1.0,
            0.5 * (warn_hours / total_points) + 0.5 * (crit_hours / total_points),
        )
        risk_level = classify_risk_level(risk_prob)

        output_rows.append(
            {
                "product_type": product_name,
                "container_number": container_id,
                "start_time": group["prediction_time"].min(),
                "end_time": group["prediction_time"].max(),
                "num_points": len(group),
                "temp_warn_hours": int(temp_warn_mask.sum()),
                "temp_crit_hours": int(temp_crit_mask.sum()),
                "hum_warn_hours": int(hum_warn_mask.sum()),
                "hum_crit_hours": int(hum_crit_mask.sum()),
                "warn_hours": warn_hours,
                "crit_hours": crit_hours,
                "longest_warn_run": longest_warn_run,
                "longest_crit_run": longest_crit_run,
                "risk_probability": round(risk_prob, 4),
                "risk_level": risk_level,
                "mean_temp_pred": round(group["temperature_pred"].mean(), 3),
                "mean_humidity_pred": round(group["humidity_pred"].mean(), 3),
            }
        )

        grouped_data.append((container_id, group))

    n = len(grouped_data)
    if n > 0:
        ncols = 2
        nrows = (n + 1) // 2 if n > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9))
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, (container_id, group) in zip(axes, grouped_data):
            ax2 = ax.twinx()

            ax.plot(
                group["prediction_time"],
                group["temperature_pred"],
                label="Temperature",
                linewidth=1.5,
            )
            ax2.plot(
                group["prediction_time"],
                group["humidity_pred"],
                color="red",
                label="Humidity",
                linewidth=1.5,
            )

            ax.axhline(temp_crit_high, linestyle="--", linewidth=1)
            ax.axhline(temp_crit_low, linestyle="--", linewidth=1)
            ax2.axhline(hum_crit_high, linestyle="--", linewidth=1, color="red")
            ax2.axhline(hum_crit_low, linestyle="--", linewidth=1, color="red")

            ax.set_title(
                f"{container_id} | {product_name.capitalize()} | "
                f"T crit={temp_crit_high}°C, H crit={hum_crit_high}%"
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature (°C)")
            ax2.set_ylabel("Humidity (%)")
            ax.tick_params(axis="x", rotation=45)

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(
            f"{product_name.capitalize()} Containers — Temperature & Humidity Overview",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = figure_subdir / f"{_safe_name(product_name)}_72h_overview.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    result_df = pd.DataFrame(output_rows)
    return result_df