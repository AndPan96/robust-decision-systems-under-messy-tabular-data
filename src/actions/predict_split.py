import pandas as pd
from src.data.load_data import load_dataset
from src.config.paths import DATASET_PATH, MONITORING_IDS, MONITORING_X, PROCESSED_IDS

def predict_split():

    ids, X, y = load_dataset(DATASET_PATH)

    proc_idx = pd.read_parquet(PROCESSED_IDS)["SK_ID_CURR"]
    mask_new = ~ids.isin(proc_idx)
    ids = ids[mask_new]
    X = X[mask_new]
    y = y[mask_new]

    mask = y.isna()
    ids = ids[mask]
    X = X[mask]

    X.to_parquet(MONITORING_X)
    ids.to_frame("ID").to_parquet(MONITORING_IDS)
