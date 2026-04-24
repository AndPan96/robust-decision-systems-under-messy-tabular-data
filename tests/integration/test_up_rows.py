import pandas as pd
import json
from unittest.mock import patch
from src.pipelines.up_rows import update_rows_pipeline
from src.config.paths import RAW_APPLICATION_TRAIN, UPDATES_FILE, PROCESSED_IDS, PROCESSED_PREDS, PROCESSED_X
from pathlib import Path

def test_update_rows_pipeline(set_environment, reset_environment):

    try:
        set_environment()

        df = pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3],
            "F1": [.1, .2, .3],
            "TARGET": [None, None, 1]
        })

        df.to_csv(RAW_APPLICATION_TRAIN, index=False)

        pd.DataFrame(columns=["SK_ID_CURR"]).to_parquet(PROCESSED_IDS, index=False)
        pd.DataFrame(columns=["TARGET_PRED"]).to_parquet(PROCESSED_PREDS, index=False)
        pd.DataFrame(columns=["PLACEHOLDER"]).to_parquet(PROCESSED_X, index=False)

        updates = {
            "application_train": [
                {"SK_ID_CURR": 1, "TARGET": 0},
                {"SK_ID_CURR": 3, "TARGET": 1},
                {"SK_ID_CURR": 4, "TARGET": 1}
            ]
        }

        with open(UPDATES_FILE, "w") as f:
            json.dump(updates, f)

        update_rows_pipeline()

        updated_df = pd.read_csv(RAW_APPLICATION_TRAIN)

        assert updated_df.loc[updated_df["SK_ID_CURR"] == 1, "TARGET"].iloc[0] == 0
        assert updated_df.loc[updated_df["SK_ID_CURR"] == 3, "TARGET"].iloc[0] == 1
        assert 4 in updated_df["SK_ID_CURR"].values
        assert not UPDATES_FILE.exists()

    finally:
        reset_environment()