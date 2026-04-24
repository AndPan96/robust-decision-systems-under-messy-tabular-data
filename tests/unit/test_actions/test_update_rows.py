import pytest
import pandas as pd
from unittest.mock import patch
from src.actions.update_rows import update_rows
from src.config.paths import RAW_APPLICATION_TRAIN, DATASET_PATH, PROCESSED_IDS, PROCESSED_PREDS, PROCESSED_X
from src.config.state import STATE_FILE

@pytest.fixture
def fake_ds():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [None, 1, None],
        "F1": [10, 20, 30]
    })

def test_update_rows(fake_ds: pd.DataFrame):

    updates = {
        "application_train": [
            {"SK_ID_CURR": 1, "TARGET": 0},
            {"SK_ID_CURR": 3, "TARGET": 1},
            {"SK_ID_CURR": 4, "TARGET": 1}
        ]
    }

    fake_ds.to_csv(RAW_APPLICATION_TRAIN, index=False)

    with patch("src.actions.update_rows.pd.read_parquet") as mock_read_parquet:
    
        mock_read_parquet.side_effect = [
            pd.DataFrame({"SK_ID_CURR": [1, 2, 3]}),
            pd.DataFrame({"TARGET_PRED": [1, 2, 3]}),
            pd.DataFrame({"PLACEHOLDER_X": [1, 2, 3]})
        ]
    
        update_rows(updates)

    df = pd.read_csv(RAW_APPLICATION_TRAIN)

    assert df.loc[df["SK_ID_CURR"] == 1, "TARGET"].iloc[0] == 0
    assert df.loc[df["SK_ID_CURR"] == 2, "TARGET"].iloc[0] == 1
    assert df.loc[df["SK_ID_CURR"] == 3, "TARGET"].iloc[0] == 1
    assert 4 in df["SK_ID_CURR"].values
    assert df.loc[df["SK_ID_CURR"] == 4, "TARGET"].iloc[0] == 1

    RAW_APPLICATION_TRAIN.unlink(missing_ok=True)
    DATASET_PATH.unlink(missing_ok=True)
    PROCESSED_IDS.unlink(missing_ok=True)
    PROCESSED_PREDS.unlink(missing_ok=True)
    PROCESSED_X.unlink(missing_ok=True)
    STATE_FILE.unlink(missing_ok=True)
