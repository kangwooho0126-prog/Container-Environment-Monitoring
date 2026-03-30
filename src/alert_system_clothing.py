from pathlib import Path

from alert_common import run_alert_pipeline
from utils import ensure_dir


def main():
    output_dir = Path("outputs/predictions")
    ensure_dir(output_dir)

    result_df = run_alert_pipeline(
        product_name="clothing",
        temp_warn_low=10,
        temp_warn_high=30,
        temp_crit_low=5,
        temp_crit_high=35,
        hum_warn_low=30,
        hum_warn_high=70,
        hum_crit_low=20,
        hum_crit_high=80,
        horizon=72,
    )
    result_df.to_csv(output_dir / "clothing_alert.csv", index=False)
    print("Saved: outputs/predictions/clothing_alert.csv")


if __name__ == "__main__":
    main()