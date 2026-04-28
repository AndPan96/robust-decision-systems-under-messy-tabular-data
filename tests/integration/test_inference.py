from typing import Any
import pandas as pd
from unittest.mock import patch
import torch
from src.pipelines.inference import inference_pipeline
from src.config.registry import deploy_model
from src.actions.transform_data import transform_data, FEATURE_SCHEMA
from src.config.state import load_state, save_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from src.models.linear_model import LinearModel
from src.config.paths import MONITORING_IDS, MONITORING_X, PROCESSED_PREDS, DATASET_PATH,\
        RAW_APPLICATION_TRAIN, RAW_BUREAU, RAW_BUREAU_BALANCE, RAW_CREDIT_CARD_BALANCE, \
        RAW_INSTALLMENTS_PAYMENTS, RAW_POS_CASH_BALANCE, RAW_PREVIOUS_APPLICATION, \
        PROCESSED_IDS

def test_inference_pipeline(set_environment, reset_environment):

    try:
        set_environment()

        app = pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [None, None, None],
            "CNT_CHILDREN": [1, 0, 2],
            "AMT_INCOME_TOTAL": [1000, 2000, 3000],
            "AMT_CREDIT": [5000, 6000, 7000],
            "AMT_ANNUITY": [100, 200, 300],

            "FLAG_MOBIL": [1, 1, 1],
            "FLAG_EMAIL": [0, 1, 0],
            "LIVE_CITY_NOT_WORK_CITY": [0, 0, 1],

            **{f"FLAG_DOCUMENT_{i}": [0, 0, 0] for i in range(2, 22)}
        })

        bureau = pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3],
            "SK_ID_BUREAU": [10, 20, 30],
            "DAYS_ENDDATE_FACT": [0, 0, 0],
            "CNT_CREDIT_PROLONG": [0, 0, 0],
            "AMT_CREDIT_SUM": [100, 200, 300],
            "AMT_CREDIT_SUM_DEBT": [50, 100, 150],
        })

        bb = pd.DataFrame({
            "SK_ID_BUREAU": [10, 20, 30],
            "MONTHS_BALANCE": [-1, -2, -3]
        })

        pcb = pd.DataFrame({
            "SK_ID_PREV": [100, 200, 300],
            "SK_DPD": [2, 2, 2],
            "SK_DPD_DEF": [1, 1, 1]
        })

        ip = pd.DataFrame({
            "SK_ID_PREV": [100, 200, 300],
            "DAYS_ENTRY_PAYMENT": [10, 10, 10],
            "DAYS_INSTALMENT": [5, 5, 5],
            "AMT_PAYMENT": [200, 200, 200],
            "AMT_INSTALMENT": [180, 180, 180],
        })

        ccb = pd.DataFrame({
            "SK_ID_PREV": [100, 200, 300],
            "AMT_BALANCE": [300, 300, 300],
            "SK_DPD": [2, 2, 2],
            "SK_DPD_DEF": [1, 1, 1]
        })

        pa = pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3],
            "SK_ID_PREV": [100, 200, 300],
            "AMT_ANNUITY": [100, 100, 100],
            "AMT_APPLICATION": [200, 200, 200],
            "AMT_CREDIT": [150, 150, 150],
            "AMT_DOWN_PAYMENT": [50, 50, 50],
            "NAME_YIELD_GROUP": ["middle", "middle", "middle"]
        })

        app.to_csv(RAW_APPLICATION_TRAIN, index=False)
        bureau.to_csv(RAW_BUREAU, index=False)
        bb.to_csv(RAW_BUREAU_BALANCE, index=False)
        pcb.to_csv(RAW_POS_CASH_BALANCE, index=False)
        ip.to_csv(RAW_INSTALLMENTS_PAYMENTS, index=False)
        ccb.to_csv(RAW_CREDIT_CARD_BALANCE, index=False)
        pa.to_csv(RAW_PREVIOUS_APPLICATION, index=False)

        pd.DataFrame(columns=["SK_ID_CURR"]).to_parquet(PROCESSED_IDS, index=False)

        transform_data()
        state = load_state()
        state["transform_required"] = True # making the pipeline run the transform again
        save_state(state)
        df = pd.read_csv(DATASET_PATH)
        feature_cols = (
            FEATURE_SCHEMA["mean"] +
            FEATURE_SCHEMA["zero"] +
            FEATURE_SCHEMA["binary"]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("identity", FunctionTransformer(), feature_cols)
            ],
            remainder="drop"
        )
        preprocessor.set_output(transform="pandas")
        preprocessor.fit(df[feature_cols])
        model = LinearModel(len(feature_cols), 2)
        deploy_model(model, preprocessor, {
            "model_class_name": "LinearModel", 
            "metrics": {}, 
            "input_dim": len(feature_cols),
            "report_path": "./",
            "plots_path": "./"})

        inference_pipeline()

        preds = pd.read_parquet(PROCESSED_PREDS)
        assert len(preds) == 3
        assert "TARGET_PRED" in preds.columns

        ids = pd.read_parquet(MONITORING_IDS)
        X = pd.read_parquet(MONITORING_X)
        assert len(ids) == 0 
        assert len(X) == 0
    
    finally:
        reset_environment()