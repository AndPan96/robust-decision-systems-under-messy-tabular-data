from typing import Any
import pandas as pd
from unittest.mock import patch
import torch
from src.pipelines.inference import inference_pipeline
from src.config.registry import deploy_model
from src.models.linear_model import LinearModel
from src.config.paths import DATASET_PATH, MONITORING_IDS, MONITORING_X, PROCESSED_PREDS

def test_inference_pipeline():

    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "F1": [.1, .2, .3],
        "F2": [1, 2, 3],
        "TARGET": [None, None, None]
    })

    df.to_csv(DATASET_PATH, index=False)
    model = LinearModel(2, 2)
    deploy_model(model, {"model_class_name": "LinearModel", "metrics": {}})

    inference_pipeline()

    preds = pd.read_parquet(PROCESSED_PREDS)
    assert len(preds) == 3
    assert "TARGET_PRED" in preds.columns

    ids = pd.read_parquet(MONITORING_IDS)
    X = pd.read_parquet(MONITORING_X)
    assert len(ids) == 0 
    assert len(X) == 0