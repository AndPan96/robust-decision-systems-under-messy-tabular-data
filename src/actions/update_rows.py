import pandas as pd
from src.config.paths import DATASET_PATH


def update_rows(updates: list[dict]):

    if len(updates) == 0:
        return
    
    df = pd.read_csv(DATASET_PATH)

    updates_df = pd.DataFrame(updates)

    df = df.set_index("SK_ID_CURR")
    updates_df = updates_df.set_index("SK_ID_CURR")

    for idx, row in updates_df.iterrows():

        if idx not in df.index:
            continue

        if pd.notna(df.at[idx, "TARGET"]):
            continue

        for col, value in row.items():
            df.loc[idx, col] = value

    df.reset_index(inplace=True)

    df.to_csv(DATASET_PATH, index=False)