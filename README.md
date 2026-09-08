# Maritime Vessel Anomaly Detection

An interpretable machine-learning baseline for detecting unusual vessel behavior from operational and environmental signals.

## Why this project

Shipping systems produce many signals at once: speed, course changes, route distance, draught, weather, visibility, traffic density, and AIS signal gaps. This project turns those signals into a reproducible anomaly-detection workflow that can be inspected, tested, and extended.

## Dataset

The included synthetic dataset contains 6,000 vessel observations with:

- vessel type and seasonality
- speed, course-change angle, route distance, and draught
- wind, visibility, and traffic density
- AIS signal-gap duration
- binary `anomaly` target

The data is synthetic and is intended for engineering demonstration and model prototyping, not operational decisions.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

The training script prints a classification report and saves a feature-importance chart to `outputs/feature_importance.png`.

## Workflow

1. Load and validate the tabular data.
2. One-hot encode categorical vessel type.
3. Train a reproducible Random Forest baseline.
4. Apply a documented screening threshold and report precision, recall, F1, ROC-AUC, and confusion matrix.
5. Export model diagnostics for inspection.

## Engineering principles

- reproducible random seed
- explicit feature list
- no hidden network calls
- validation before training
- clear separation between data, source code, and outputs

## Next improvements

- compare calibrated tree models and gradient boosting
- add temporal and vessel-level split strategies
- evaluate class imbalance and threshold selection
- connect the feature pipeline to live AIS data only after source and geofence validation

## License

MIT
