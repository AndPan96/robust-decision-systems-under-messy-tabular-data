import pytest
import pandas as pd
from unittest.mock import patch
from src.actions.monitor_model import monitor_model
from src.config.paths import PROCESSED_PREDS, DATASET_PATH

@pytest.fixture
def fake_data():

    pred = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5],
        "TARGET_PRED": [0, 1, 0, 1, 0]
    })

    raw = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5],
        "TARGET": [0, 1, 1, 1, 0]
    })

    return pred, raw

def test_monitor_model_threshold(fake_data):

    preds: pd.DataFrame
    raw: pd.DataFrame
    preds, raw = fake_data

    preds.to_parquet(PROCESSED_PREDS, index=False)
    raw.to_parquet(DATASET_PATH, index=False)

    with patch("src.actions.monitor_model.MONITOR_WINDOW", 3):

        result = monitor_model()

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "retrain" in result

    preds.to_parquet(PROCESSED_PREDS, index=False)
    raw.to_parquet(DATASET_PATH, index=False)

    with patch("src.actions.monitor_model.MONITOR_WINDOW", 10):

        result = monitor_model()

        assert result is None