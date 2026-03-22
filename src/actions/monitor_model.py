import pandas as pd
from src.config.data_config import RETRAIN_THRESHOLD, MONITOR_WINDOW
from src.config.paths import PROCESSED_PREDS, DATASET_PATH

def monitor_model():

    preds = pd.read_parquet(PROCESSED_PREDS)
    raw = pd.read_parquet(DATASET_PATH)

    labels = raw[["SK_ID_CURR", "TARGET"]]

    df = preds.merge(labels, on="SK_ID_CURR", how="left")

    df = df.dropna(subset=["TARGET"])
    
    if len(df) < MONITOR_WINDOW:
        return None
    
    df = df.sort_values("SK_ID_CURR").tail(MONITOR_WINDOW)

    acc = (df["TARGET_PRED"] == df["TARGET"]).mean()

    return {
        "accuracy": acc,
        "retrain": (acc < RETRAIN_THRESHOLD)
    }
