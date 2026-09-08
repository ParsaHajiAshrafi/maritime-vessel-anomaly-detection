from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "maritime_vessel_anomaly_synthetic.csv"


def test_dataset_schema_and_size():
    frame = pd.read_csv(DATA_PATH)
    expected = {
        "vessel_id", "vessel_type", "month", "speed_knots",
        "course_change_deg", "distance_to_route_nm", "draught_m",
        "wind_speed_knots", "visibility_nm", "traffic_density",
        "signal_gap_minutes", "anomaly",
    }
    assert expected.issubset(frame.columns)
    assert len(frame) == 6000


def test_dataset_contains_both_classes():
    frame = pd.read_csv(DATA_PATH)
    assert set(frame["anomaly"].unique()) == {0, 1}
