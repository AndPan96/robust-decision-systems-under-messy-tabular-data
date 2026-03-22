import pytest
import pandas as pd
from unittest.mock import patch
from src.actions.update_rows import update_rows
from src.config.paths import DATASET_PATH

@pytest.fixture
def fake_ds():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [None, 1, None],
        "F1": [10, 20, 30]
    })

def test_update_rows(fake_ds: pd.DataFrame):

    updates = [
        {"SK_ID_CURR": 1 , "TARGET": 0},
        {"SK_ID_CURR": 3 , "TARGET": 1},
        {"SK_ID_CURR": 4 , "TARGET": 1}
    ]

    fake_ds.to_csv(DATASET_PATH, index=False)

    update_rows(updates)

    df = pd.read_csv(DATASET_PATH)

    assert df.loc[df["SK_ID_CURR"] == 1, "TARGET"].iloc[0] == 0
    assert df.loc[df["SK_ID_CURR"] == 2, "TARGET"].iloc[0] == 1
    assert df.loc[df["SK_ID_CURR"] == 3, "TARGET"].iloc[0] == 1


