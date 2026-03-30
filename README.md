# Container Environment Monitoring and Early Warning System

## Project Overview

This project builds a container environment monitoring pipeline for **temperature and humidity prediction** and a **product-specific early warning system**.

The workflow is:

1. Train forecasting models for temperature and humidity
2. Generate prediction results for multiple containers
3. Evaluate prediction performance
4. Apply product-specific threshold rules
5. Produce 72H early warning summaries and overview figures

The system is designed for multi-container monitoring scenarios such as cold chain logistics, sensitive cargo handling, and environment-controlled transportation.

---

## Project Structure

```text
container-environment-monitoring/
├─ data/
├─ models/
├─ notebooks/
├─ outputs/
│  ├─ figures/
│  │  ├─ clothing/
│  │  ├─ electronics/
│  │  ├─ evaluation/
│  │  └─ food/
│  └─ predictions/
├─ src/
│  ├─ alert_common.py
│  ├─ alert_system_clothing.py
│  ├─ alert_system_electronics.py
│  ├─ alert_system_food.py
│  ├─ config.py
│  ├─ evaluate.py
│  ├─ plot_results.py
│  ├─ predict.py
│  ├─ train.py
│  └─ utils.py
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

## Main Functions

### 1. Temperature and Humidity Prediction
The project trains predictive models for container temperature and humidity and saves:

- trained models
- prediction results
- evaluation metrics
- container-level metrics

Main outputs:

- `outputs/predictions/final_pred_temp_hum.csv`
- `outputs/predictions/final_pred_temp_hum.xlsx`
- `outputs/predictions/metrics.csv`
- `outputs/predictions/container_metrics.csv`

---

### 2. Multi-Container Evaluation Visualization
Prediction results are visualized at the container level to compare **true vs predicted trends** across multiple containers.

Main output:

- `outputs/figures/evaluation/`

Typical figures include:

- temperature comparison by container
- humidity comparison by container

---

### 3. Product-Specific Early Warning System
The early warning system directly uses prediction results from four containers and applies different risk thresholds for:

- clothing
- electronics
- food

Main scripts:

- `src/alert_system_clothing.py`
- `src/alert_system_electronics.py`
- `src/alert_system_food.py`

Main outputs:

- `outputs/predictions/clothing_alert.csv`
- `outputs/predictions/electronics_alert.csv`
- `outputs/predictions/food_alert.csv`

Overview figures:

- `outputs/figures/clothing/clothing_72h_overview.png`
- `outputs/figures/electronics/electronics_72h_overview.png`
- `outputs/figures/food/food_72h_overview.png`

Each overview figure combines **four containers in one dashboard** and shows:

- predicted temperature curve
- predicted humidity curve
- product-specific threshold lines
- 72-hour monitoring window

---

## Model Performance

The current prediction metrics are:

| Target | MAE | RMSE | R² | safeMAPE | sMAPE | WAPE |
|---|---:|---:|---:|---:|---:|---:|
| Temperature | 3.9001 | 4.3957 | 0.6510 | 83.7260% | 53.2499% | 37.3074% |
| Humidity | 8.7483 | 9.9609 | 0.6965 | 26.6004% | 32.6904% | 22.2807% |

These results show that:

- the model captures the overall temperature trend reasonably well
- humidity prediction remains more challenging, but still provides useful early warning signals
- the project is not only a forecasting task, but also a **decision-support monitoring system**

---

## Early Warning Logic

For each product type, the system:

1. groups prediction results by `container_number`
2. extracts the most recent **72 prediction points** for each container
3. checks whether predicted temperature and humidity exceed warning / critical thresholds
4. calculates summary indicators such as:
   - warning hours
   - critical hours
   - longest warning run
   - longest critical run
   - risk probability
   - risk level

This design makes the project more practical than a pure forecasting demo, because the output can directly support monitoring decisions.

---

## Example Monitoring Scenarios

### Clothing
Used for products sensitive to heat and excessive humidity, with threshold-based monitoring for storage stability.

### Electronics
Used for products requiring tighter environmental control, especially for humidity-related risk.

### Food
Used for products with stricter temperature and humidity safety requirements over the monitoring horizon.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python src/train.py
```

### 3. Generate evaluation figures

```bash
python src/plot_results.py
```

### 4. Run early warning scripts

```bash
python src/alert_system_clothing.py
python src/alert_system_electronics.py
python src/alert_system_food.py
```

---

## Key Features

- multi-container forecasting workflow
- temperature and humidity dual-target monitoring
- product-specific early warning rules
- 72H overview visualization
- container-level evaluation outputs
- practical risk-monitoring orientation

---

## Notes

This repository focuses on an **end-to-end environment monitoring pipeline** rather than only improving raw prediction accuracy.  
The main contribution is combining:

- forecasting
- multi-container visualization
- threshold-based alerting
- product-specific risk interpretation

into one unified project.
