import re
from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    DATA_FILE,
    DEFAULT_FORECAST_HOURS,
    FORECAST_RESULT_FILE,
    HUM_MODEL_FILE,
    HUM_TGT,
    PRED_DIR,
    TEMP_MODEL_FILE,
    TEMP_TGT,
)
from train import build_feature_columns, load_data, split_by_container
from utils import ensure_dir


np.random.seed(42)
tf.random.set_seed(42)


def _parse_lag_columns(columns, prefix):
    mapping = {}
    pattern = re.compile(rf"^{re.escape(prefix)}_lag(\d+)$")
    for col in columns:
        m = pattern.match(col)
        if m:
            mapping[int(m.group(1))] = col
    return mapping


def _parse_window_columns(columns, prefix):
    mapping = {}
    pattern = re.compile(rf"^{re.escape(prefix)}_(mean|std|max|min)(\d+)$")
    for col in columns:
        m = pattern.match(col)
        if m:
            stat = m.group(1)
            window = int(m.group(2))
            mapping[(stat, window)] = col
    return mapping


def _hour_features(ts: pd.Timestamp):
    hour = int(ts.hour)
    dow = int(ts.dayofweek)
    month = int(ts.month)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    return {
        "hour": hour,
        "dow": dow,
        "month": month,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }


def _safe_stat(values, stat_name):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    if stat_name == "mean":
        return float(np.mean(arr))
    if stat_name == "std":
        return float(np.std(arr))
    if stat_name == "max":
        return float(np.max(arr))
    if stat_name == "min":
        return float(np.min(arr))
    raise ValueError(f"Unsupported stat_name: {stat_name}")


def _infer_time_step_hours(frame, time_col):
    if time_col is None or time_col not in frame.columns:
        return 1
    diffs = (
        frame.sort_values(time_col)[time_col]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(3600.0)
    )
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1
    return max(1, int(round(float(diffs.median()))))


def _compute_feature_stats(df, feature_cols):
    train_df, _, _, _, _ = split_by_container(df)
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0).fillna(1.0)
    return mean, std


def _build_future_row(base_row, next_time, state):
    row = base_row.copy()

    row["time"] = next_time
    row["time_hour"] = next_time.floor("h")

    for key, value in _hour_features(next_time).items():
        if key in row.index:
            row[key] = value

    temp_hist = list(state["temperature_hist"])
    hum_hist = list(state["humidity_hist"])
    temp_out_hist = list(state["temp_out_hist"])
    hum_out_hist = list(state["hum_out_hist"])

    if "temperature" in row.index:
        row["temperature"] = temp_hist[-1]
    if "humidity" in row.index:
        row["humidity"] = hum_hist[-1]
    if "temp_out" in row.index:
        row["temp_out"] = temp_out_hist[-1]
    if "hum_out" in row.index:
        row["hum_out"] = hum_out_hist[-1]

    feature_groups = [
        ("temperature", temp_hist),
        ("humidity", hum_hist),
        ("temp_out", temp_out_hist),
        ("hum_out", hum_out_hist),
    ]

    for prefix, hist in feature_groups:
        lag_map = state["lag_maps"][prefix]
        for lag, col in lag_map.items():
            row[col] = hist[-lag] if len(hist) >= lag else np.nan

        window_map = state["window_maps"][prefix]
        for (stat_name, window), col in window_map.items():
            values = hist[-window:] if len(hist) >= window else hist[:]
            row[col] = _safe_stat(values, stat_name)

    if TEMP_TGT in row.index:
        row[TEMP_TGT] = np.nan
    if HUM_TGT in row.index:
        row[HUM_TGT] = np.nan

    return row


def _predict_one(row, model, feature_cols, mean, std):
    x = ((row[feature_cols] - mean[feature_cols]) / std[feature_cols]).fillna(0.0)
    x = x.to_numpy(dtype=np.float32).reshape(1, -1)
    pred = model.predict(x, verbose=0).reshape(-1)[0]
    return float(pred)


def main():
    ensure_dir(PRED_DIR)

    df = load_data(DATA_FILE)
    _, _, _, cid_col, time_col = split_by_container(df)

    if cid_col is None:
        raise ValueError("No container id column found.")
    if time_col is None:
        raise ValueError("No time/date column found.")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()

    temp_feature_cols = build_feature_columns(df, "temperature")
    hum_feature_cols = build_feature_columns(df, "humidity")
    temp_mean, temp_std = _compute_feature_stats(df, temp_feature_cols)
    hum_mean, hum_std = _compute_feature_stats(df, hum_feature_cols)

    temp_model = tf.keras.models.load_model(TEMP_MODEL_FILE, compile=False)
    hum_model = tf.keras.models.load_model(HUM_MODEL_FILE, compile=False)

    all_columns = list(df.columns)
    lag_maps = {
        "temperature": _parse_lag_columns(all_columns, "temperature"),
        "humidity": _parse_lag_columns(all_columns, "humidity"),
        "temp_out": _parse_lag_columns(all_columns, "temp_out"),
        "hum_out": _parse_lag_columns(all_columns, "hum_out"),
    }
    window_maps = {
        "temperature": _parse_window_columns(all_columns, "temperature"),
        "humidity": _parse_window_columns(all_columns, "humidity"),
        "temp_out": _parse_window_columns(all_columns, "temp_out"),
        "hum_out": _parse_window_columns(all_columns, "hum_out"),
    }

    results = []

    for cid, g in df.groupby(cid_col):
        g = g.sort_values(time_col).reset_index(drop=True)
        if g.empty:
            continue

        base_row = g.iloc[-1].copy()
        step_hours = _infer_time_step_hours(g, time_col)

        state = {
            "temperature_hist": deque(g["temperature"].astype(float).tail(48).tolist(), maxlen=256),
            "humidity_hist": deque(g["humidity"].astype(float).tail(48).tolist(), maxlen=256),
            "temp_out_hist": deque(g["temp_out"].astype(float).tail(48).tolist(), maxlen=256),
            "hum_out_hist": deque(g["hum_out"].astype(float).tail(48).tolist(), maxlen=256),
            "lag_maps": lag_maps,
            "window_maps": window_maps,
        }

        current_time = pd.to_datetime(base_row[time_col])

        for horizon in range(1, DEFAULT_FORECAST_HOURS + 1):
            next_time = current_time + pd.Timedelta(hours=step_hours)
            future_row = _build_future_row(base_row, next_time, state)

            temp_pred = _predict_one(future_row, temp_model, temp_feature_cols, temp_mean, temp_std)
            hum_pred = _predict_one(future_row, hum_model, hum_feature_cols, hum_mean, hum_std)

            results.append(
                {
                    "container_number": cid,
                    "forecast_hour": horizon,
                    "prediction_time": next_time,
                    "temperature_pred": round(temp_pred, 6),
                    "humidity_pred": round(hum_pred, 6),
                }
            )

            state["temperature_hist"].append(temp_pred)
            state["humidity_hist"].append(hum_pred)
            state["temp_out_hist"].append(float(state["temp_out_hist"][-1]))
            state["hum_out_hist"].append(float(state["hum_out_hist"][-1]))
            current_time = next_time
            base_row = future_row

    forecast_df = pd.DataFrame(results)
    forecast_df.to_csv(FORECAST_RESULT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved 72h forecast to: {FORECAST_RESULT_FILE}")
    print(forecast_df.head())
    if not forecast_df.empty:
        print("Rows per container:")
        print(forecast_df.groupby("container_number").size())


if __name__ == "__main__":
    main()
