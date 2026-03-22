import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    ids = df["SK_ID_CURR"]
    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    y = df["TARGET"]

    return ids, X, y

