import pandas as pd
import json
from unittest.mock import patch
from src.pipelines.update_rows import update_rows_pipeline
from src.config.paths import DATASET_PATH, UPDATES_FILE
from pathlib import Path

def test_update_rows_pipeline():

    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "F1": [.1, .2, .3],
        "TARGET": [None, None, 1]
    })

    df.to_csv(DATASET_PATH, index=False)

    updates = [
        {"SK_ID_CURR": 1, "TARGET": 0},
        {"SK_ID_CURR": 3, "TARGET": 0},
        {"SK_ID_CURR": 999, "TARGET": 1}
    ]

    with open(UPDATES_FILE, "w") as f:
        json.dump(updates, f)

    update_rows_pipeline()

    updated_df = pd.read_csv(DATASET_PATH)

    assert updated_df.loc[updated_df["SK_ID_CURR"] == 1, "TARGET"].iloc[0] == 0
    assert updated_df.loc[updated_df["SK_ID_CURR"] == 3, "TARGET"].iloc[0] == 1
    assert not UPDATES_FILE.exists()