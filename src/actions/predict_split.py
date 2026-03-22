import pandas as pd
from src.data.load_data import load_dataset
import json
from src.config.paths import STATE_PATH, DATASET_PATH, MONITORING_IDS, MONITORING_X

def load_checkpoint():

    if not STATE_PATH.exists():
        return None
    
    with open(STATE_PATH, "r") as f:
        state = json.load(f)

    return state["last_index"]

def save_checkpoint(idx):

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(STATE_PATH, "w") as f:
        json.dump({"last_index": int(idx)}, f)

def predict_split():

    ids, X, y = load_dataset(DATASET_PATH)

    last_idx = load_checkpoint()
    if last_idx is not None:
        mask_new = ids > last_idx
        ids = ids[mask_new]
        X = X[mask_new]
        y = y[mask_new]

    mask = y.isna()
    ids = ids[mask]
    X = X[mask]

    X.to_parquet(MONITORING_X)
    ids.to_frame("ID").to_parquet(MONITORING_IDS)

    if len(ids) > 0:
        save_checkpoint(ids.max())