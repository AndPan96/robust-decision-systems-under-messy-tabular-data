import pytest
import pandas as pd
from unittest.mock import patch
from src.actions.monitor_model import monitor_model
from src.config.paths import PROCESSED_IDS, PROCESSED_PREDS, PROCESSED_X, \
    DATASET_PATH, MONITORING_IDS, MONITORING_X

@pytest.fixture
def fake_data():

    pred = pd.DataFrame({
        "TARGET_PRED": [0, 1, 0, 1, 0]
    })

    ids = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5]
    })

    X = pd.DataFrame({
        "F1": [.1, .2, .3, .4, .5],
        "F2": [1, 2, 3, 4, 5]
    })

    raw = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5],
        "TARGET": [0, 1, 1, 1, 0]
    })

    return pred, ids, X, raw

def test_monitor_model_threshold(fake_data):

    preds: pd.DataFrame
    ids: pd.DataFrame
    X: pd.DataFrame
    raw: pd.DataFrame
    preds, ids, X, raw = fake_data

    preds.to_parquet(PROCESSED_PREDS, index=False)
    ids.to_parquet(PROCESSED_IDS, index=False)
    X.to_parquet(PROCESSED_X, index=False)
    raw.to_csv(DATASET_PATH, index=False)

    with patch("src.actions.monitor_model.MONITOR_WINDOW", 3):

        result = monitor_model()

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "retrain" in result

    preds.to_parquet(PROCESSED_PREDS, index=False)
    raw.to_csv(DATASET_PATH, index=False)

    with patch("src.actions.monitor_model.MONITOR_WINDOW", 10):

        result = monitor_model()

        assert result is None

    PROCESSED_PREDS.unlink(missing_ok=True)
    PROCESSED_IDS.unlink(missing_ok=True)
    PROCESSED_X.unlink(missing_ok=True)
    DATASET_PATH.unlink(missing_ok=True)
    MONITORING_IDS.unlink(missing_ok=True)
    MONITORING_X.unlink(missing_ok=True)