import pandas as pd
from src.config.paths import MONITORING_IDS, MONITORING_X, PROCESSED_IDS, PROCESSED_X, PROCESSED_PREDS
from src.config.registry import load_current_model, DEVICE
from pathlib import Path
import torch
from typing import cast

def append_parquet(path: Path, df: pd.DataFrame):

    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)

    df.to_parquet(path)


def predict():

    ids = pd.read_parquet(MONITORING_IDS)
    if len(ids) == 0:
        return []
    X = pd.read_parquet(MONITORING_X)

    model, preprocessor, _ = load_current_model()
    X_imp = cast(pd.DataFrame, preprocessor.transform(X))

    with torch.no_grad():
        X_tensor = torch.tensor(X_imp.values, dtype=torch.float32).to(DEVICE)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    preds = pd.DataFrame({"TARGET_PRED": preds})

    append_parquet(PROCESSED_IDS, ids)
    append_parquet(PROCESSED_X, X_imp)
    append_parquet(PROCESSED_PREDS, preds)

    ids.iloc[0:0].to_parquet(MONITORING_IDS)
    X_imp.iloc[0:0].to_parquet(MONITORING_X)

    return preds