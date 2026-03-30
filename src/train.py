import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, callbacks

from config import (
    DATA_FILE, MODEL_DIR, PRED_DIR,
    TEMP_MODEL_FILE, HUM_MODEL_FILE, PRED_RESULT_FILE,
    BATCH_SIZE, EPOCHS,
    TEMP_TGT, HUM_TGT
)
from utils import ensure_dir, norm_col

np.random.seed(42)
tf.random.set_seed(42)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-6)


def safe_mape(y_true, y_pred, threshold=1.0):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-6
    if not mask.any():
        return np.nan
    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def wape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / (denom + 1e-6) * 100


def load_data(path):
    path = str(path)
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [norm_col(c) for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True)).fillna(0)
    return df


def split_by_container(frame):
    """
    Current logic:
    - Use ALL containers
    - Sort each container by time
    - Split each container into 70% train / 20% val / 10% test

    Note:
    This is NOT a fixed 72-hour forecasting split.
    """
    cid_col = next((c for c in frame.columns if "container" in c.lower()), None)
    time_col = next(
        (c for c in frame.columns if "time" in c.lower() or "date" in c.lower()),
        None
    )

    if cid_col is None:
        raise ValueError("No container id column found.")

    train_parts, val_parts, test_parts = [], [], []

    for _, g in frame.groupby(cid_col):
        if time_col:
            g = g.sort_values(time_col)
        n = len(g)
        i_tr, i_va = int(n * 0.7), int(n * 0.9)

        train_parts.append(g.iloc[:i_tr])
        val_parts.append(g.iloc[i_tr:i_va])
        test_parts.append(g.iloc[i_va:])

    df_tr = pd.concat(train_parts).reset_index(drop=True)
    df_va = pd.concat(val_parts).reset_index(drop=True)
    df_te = pd.concat(test_parts).reset_index(drop=True)

    return df_tr, df_va, df_te, cid_col, time_col


def build_feature_columns(df, target_name):
    exclude_cols = {
        TEMP_TGT, HUM_TGT,
        "container_number", "container_number_", "container", "condition",
        "time", "address", "source_file", "country",
        "latitude", "longitude", "time_hour"
    }

    if target_name == "temperature":
        exclude_cols.update({"dow", "month"})

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    if not feature_cols:
        raise ValueError(f"No numeric feature columns found for target={target_name}.")

    return feature_cols


def standardize_x(train_df, val_df, test_df, feature_cols):
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)

    xtr = ((train_df[feature_cols] - mean) / std).values.astype(np.float32)
    xva = ((val_df[feature_cols] - mean) / std).values.astype(np.float32)
    xte = ((test_df[feature_cols] - mean) / std).values.astype(np.float32)

    return xtr, xva, xte


def standardize_y(y_train, y_val, y_test):
    mean = np.mean(y_train)
    std = np.std(y_train)
    if std < 1e-8:
        std = 1.0

    ytr = ((y_train - mean) / std).astype(np.float32)
    yva = ((y_val - mean) / std).astype(np.float32)
    yte = ((y_test - mean) / std).astype(np.float32)

    return ytr, yva, yte, mean, std


def inverse_y(y_scaled, mean, std):
    return y_scaled * std + mean


def temp_model_builder(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, name="temperature_pred")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model


def hum_model_builder(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, name="humidity_pred")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model


def main():
    ensure_dir(MODEL_DIR)
    ensure_dir(PRED_DIR)

    df = load_data(DATA_FILE)
    df_tr, df_va, df_te, cid_col, time_col = split_by_container(df)

    temp_feature_cols = build_feature_columns(df, "temperature")
    hum_feature_cols = build_feature_columns(df, "humidity")

    Xtr_t, Xva_t, Xte_t = standardize_x(df_tr, df_va, df_te, temp_feature_cols)
    Xtr_h, Xva_h, Xte_h = standardize_x(df_tr, df_va, df_te, hum_feature_cols)

    ytr_t_raw = df_tr[TEMP_TGT].values.astype(np.float32)
    yva_t_raw = df_va[TEMP_TGT].values.astype(np.float32)
    yte_t_raw = df_te[TEMP_TGT].values.astype(np.float32)

    ytr_h_raw = df_tr[HUM_TGT].values.astype(np.float32)
    yva_h_raw = df_va[HUM_TGT].values.astype(np.float32)
    yte_h_raw = df_te[HUM_TGT].values.astype(np.float32)

    ytr_t, yva_t, yte_t, temp_mean, temp_std = standardize_y(ytr_t_raw, yva_t_raw, yte_t_raw)
    ytr_h, yva_h, yte_h, hum_mean, hum_std = standardize_y(ytr_h_raw, yva_h_raw, yte_h_raw)

    t_model = temp_model_builder(Xtr_t.shape[1])
    h_model = hum_model_builder(Xtr_h.shape[1])

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
    ]

    t_model.fit(
        Xtr_t, ytr_t,
        validation_data=(Xva_t, yva_t),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    h_model.fit(
        Xtr_h, ytr_h,
        validation_data=(Xva_h, yva_h),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    y_pred_t_scaled = t_model.predict(Xte_t, verbose=0).reshape(-1)
    y_pred_h_scaled = h_model.predict(Xte_h, verbose=0).reshape(-1)

    y_pred_t = inverse_y(y_pred_t_scaled, temp_mean, temp_std)
    y_pred_h = inverse_y(y_pred_h_scaled, hum_mean, hum_std)

    t_model.save(TEMP_MODEL_FILE)
    h_model.save(HUM_MODEL_FILE)

    mae_t = np.mean(np.abs(yte_t_raw - y_pred_t))
    rmse_t = np.sqrt(np.mean((yte_t_raw - y_pred_t) ** 2))
    r2_t = r2_score(yte_t_raw, y_pred_t)
    safe_mape_t = safe_mape(yte_t_raw, y_pred_t)
    smape_t = smape(yte_t_raw, y_pred_t)
    wape_t = wape(yte_t_raw, y_pred_t)

    mae_h = np.mean(np.abs(yte_h_raw - y_pred_h))
    rmse_h = np.sqrt(np.mean((yte_h_raw - y_pred_h) ** 2))
    r2_h = r2_score(yte_h_raw, y_pred_h)
    safe_mape_h = safe_mape(yte_h_raw, y_pred_h)
    smape_h = smape(yte_h_raw, y_pred_h)
    wape_h = wape(yte_h_raw, y_pred_h)

    print(
        f"Temperature - MAE:{mae_t:.3f}, RMSE:{rmse_t:.3f}, "
        f"R2:{r2_t:.3f}, safeMAPE:{safe_mape_t:.2f}%, "
        f"sMAPE:{smape_t:.2f}%, WAPE:{wape_t:.2f}%"
    )
    print(
        f"Humidity    - MAE:{mae_h:.3f}, RMSE:{rmse_h:.3f}, "
        f"R2:{r2_h:.3f}, safeMAPE:{safe_mape_h:.2f}%, "
        f"sMAPE:{smape_h:.2f}%, WAPE:{wape_h:.2f}%"
    )

    out = pd.DataFrame({
        "container_number": df_te[cid_col].values,
        "temperature_true": yte_t_raw,
        "temperature_pred": y_pred_t,
        "humidity_true": yte_h_raw,
        "humidity_pred": y_pred_h
    })

    if time_col and time_col in df_te.columns:
        out.insert(1, "prediction_time", df_te[time_col].values)

    metrics_df = pd.DataFrame([
        {
            "target": "temperature",
            "mae": mae_t,
            "rmse": rmse_t,
            "r2": r2_t,
            "safe_mape": safe_mape_t,
            "smape": smape_t,
            "wape": wape_t
        },
        {
            "target": "humidity",
            "mae": mae_h,
            "rmse": rmse_h,
            "r2": r2_h,
            "safe_mape": safe_mape_h,
            "smape": smape_h,
            "wape": wape_h
        }
    ])

    metrics_path = PRED_DIR / "metrics.csv"
    container_metrics_path = PRED_DIR / "container_metrics.csv"
    csv_path = PRED_DIR / "final_pred_temp_hum.csv"
    xlsx_path = PRED_RESULT_FILE

    metrics_df.to_csv(metrics_path, index=False)

    container_metrics = []
    for cid, g in out.groupby("container_number"):
        container_metrics.append({
            "container_number": cid,
            "temp_mae": np.mean(np.abs(g["temperature_true"] - g["temperature_pred"])),
            "hum_mae": np.mean(np.abs(g["humidity_true"] - g["humidity_pred"]))
        })

    pd.DataFrame(container_metrics).to_csv(container_metrics_path, index=False)

    out.to_excel(xlsx_path, index=False)
    out.to_csv(csv_path, index=False)

    print("ROOT_DIR:", Path(__file__).resolve().parent.parent)
    print("PRED_DIR:", PRED_DIR)
    print("PRED_RESULT_FILE:", PRED_RESULT_FILE)
    print("Saved metrics to:", metrics_path)
    print("Saved container metrics to:", container_metrics_path)
    print("Saved csv to:", csv_path)
    print("Saved xlsx to:", xlsx_path)
    print("Files should be saved now.")


if __name__ == "__main__":
    main()