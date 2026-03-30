from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
PRED_DIR = OUTPUT_DIR / "predictions"

DATA_FILE = DATA_DIR / "China,Australia,Korea sensor data.csv.xlsx"

TEMP_MODEL_FILE = MODEL_DIR / "temp_model.keras"
HUM_MODEL_FILE = MODEL_DIR / "hum_model.keras"
SCALER_FILE = MODEL_DIR / "scalers.npz"

PRED_RESULT_FILE = PRED_DIR / "final_pred_temp_hum.xlsx"
PRED_RESULT_CSV = PRED_DIR / "final_pred_temp_hum.csv"
FORECAST_RESULT_FILE = PRED_DIR / "forecast_72h_results.csv"

DEFAULT_FORECAST_HOURS = 72

BATCH_SIZE = 64
EPOCHS = 120

TEMP_TGT = "temperature_tgt"
HUM_TGT = "humidity_tgt"

TEMP_LAGS = []
HUM_LAGS = []
SEQ_TEMPLATES = []
