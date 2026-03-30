from pathlib import Path

from alert_common import run_alert_pipeline
from utils import ensure_dir


def main():
    output_dir = Path("outputs/predictions")
    ensure_dir(output_dir)

    result_df = run_alert_pipeline(
        product_name="electronics",
        temp_warn_low=15,
        temp_warn_high=28,
        temp_crit_low=10,
        temp_crit_high=32,
        hum_warn_low=35,
        hum_warn_high=60,
        hum_crit_low=25,
        hum_crit_high=70,
        horizon=72,
    )
    result_df.to_csv(output_dir / "electronics_alert.csv", index=False)
    print("Saved: outputs/predictions/electronics_alert.csv")


if __name__ == "__main__":
    main()