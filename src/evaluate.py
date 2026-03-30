import numpy as np
import pandas as pd

from config import EVALUATION_FILE, PRED_RESULT_FILE
from train import r2_score, safe_mape, smape, wape


def metric_row(target_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "target": target_name,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "safe_mape": float(safe_mape(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
        "wape": float(wape(y_true, y_pred)),
    }


def main():
    df = pd.read_csv(PRED_RESULT_FILE)

    required = {
        "temperature_true",
        "temperature_pred",
        "humidity_true",
        "humidity_pred",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in prediction file: {sorted(missing)}")

    metrics_df = pd.DataFrame(
        [
            metric_row("temperature", df["temperature_true"].values, df["temperature_pred"].values),
            metric_row("humidity", df["humidity_true"].values, df["humidity_pred"].values),
        ]
    )
    metrics_df.to_csv(EVALUATION_FILE, index=False)
    print(metrics_df)


if __name__ == "__main__":
    main()
