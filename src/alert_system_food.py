from pathlib import Path

from alert_common import run_alert_pipeline
from utils import ensure_dir


def main():
    output_dir = Path("outputs/predictions")
    ensure_dir(output_dir)

    result_df = run_alert_pipeline(
        product_name="food",
        temp_warn_low=0,
        temp_warn_high=10,
        temp_crit_low=-2,
        temp_crit_high=15,
        hum_warn_low=50,
        hum_warn_high=75,
        hum_crit_low=40,
        hum_crit_high=85,
        horizon=72,
    )
    result_df.to_csv(output_dir / "food_alert.csv", index=False)
    print("Saved: outputs/predictions/food_alert.csv")


if __name__ == "__main__":
    main()