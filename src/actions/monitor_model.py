import pandas as pd
from src.config.data_config import RETRAIN_THRESHOLD, MONITOR_WINDOW
from src.config.paths import PROCESSED_IDS, PROCESSED_PREDS, PROCESSED_X, DATASET_PATH

def monitor_model():

    ids = pd.read_parquet(PROCESSED_IDS)
    preds = pd.read_parquet(PROCESSED_PREDS)
    raw = pd.read_csv(DATASET_PATH)

    preds = pd.concat([ids.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    labels = raw[["SK_ID_CURR", "TARGET"]]

    df = preds.merge(labels, on="SK_ID_CURR", how="left")

    df = df.dropna(subset=["TARGET"])

    if len(df) < MONITOR_WINDOW:
        return None
    
    df = df.sort_values("SK_ID_CURR").tail(MONITOR_WINDOW)

    acc = (df["TARGET_PRED"] == df["TARGET"]).mean()
    res = (acc < RETRAIN_THRESHOLD)

    if res:
        pd.read_parquet(PROCESSED_IDS).iloc[0:0].to_parquet(PROCESSED_IDS, index=False)
        pd.read_parquet(PROCESSED_PREDS).iloc[0:0].to_parquet(PROCESSED_PREDS, index=False)
        pd.read_parquet(PROCESSED_X).iloc[0:0].to_parquet(PROCESSED_X, index=False)


    return {
        "accuracy": acc,
        "retrain": res,
        "n_samples": len(df)
    }
