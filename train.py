from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "maritime_vessel_anomaly_synthetic.csv"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    required = {"vessel_type", "anomaly"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame["anomaly"].nunique() < 2:
        raise ValueError("Dataset must contain both target classes")
    return frame


def main() -> None:
    frame = load_data()
    X = frame.drop(columns=["vessel_id", "anomaly"])
    y = frame["anomaly"]
    categorical = ["vessel_type"]
    numeric = [column for column in X.columns if column not in categorical]

    preprocess = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
         ("numeric", "passthrough", numeric)]
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    # Anomaly screening prioritizes recall over the default 0.50 cutoff.
    decision_threshold = 0.30
    predictions = (probabilities >= decision_threshold).astype(int)

    print(f"rows={len(frame)} train={len(X_train)} test={len(X_test)}")
    print(classification_report(y_test, predictions, digits=3, zero_division=0))
    print("confusion_matrix=")
    print(confusion_matrix(y_test, predictions))
    print(f"roc_auc={roc_auc_score(y_test, probabilities):.3f}")
    print(f"decision_threshold={decision_threshold:.2f}")

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = pd.Series(classifier.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(12).sort_values()
    OUTPUT_DIR.mkdir(exist_ok=True)
    top.plot.barh(figsize=(9, 5), title="Top feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    print(f"saved={OUTPUT_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
