from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "maritime_vessel_anomaly_synthetic.csv"


def test_dataset_schema_and_size():
    frame = pd.read_csv(DATA_PATH)
    expected = {
        "vessel_id", "vessel_type", "season", "speed_knots",
        "course_change_deg", "route_distance_nm", "draught_m",
        "wind_knots", "visibility_km", "traffic_density",
        "ais_gap_minutes", "anomaly",
    }
    assert expected.issubset(frame.columns)
    assert len(frame) == 6000


def test_dataset_contains_both_classes():
    frame = pd.read_csv(DATA_PATH)
    assert set(frame["anomaly"].unique()) == {0, 1}
