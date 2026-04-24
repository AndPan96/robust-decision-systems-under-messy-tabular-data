import pandas as pd
from src.config.paths import RAW_APPLICATION_TRAIN, RAW_BUREAU, \
    RAW_BUREAU_BALANCE, RAW_CREDIT_CARD_BALANCE, RAW_INSTALLMENTS_PAYMENTS, \
    RAW_POS_CASH_BALANCE, RAW_PREVIOUS_APPLICATION, \
    PROCESSED_IDS, PROCESSED_PREDS, PROCESSED_X
from src.config.state import load_state, save_state

TABLE_PATHS = {
    "application_train": RAW_APPLICATION_TRAIN,
    "bureau": RAW_BUREAU,
    "bureau_balance": RAW_BUREAU_BALANCE,
    "credit_card_balance": RAW_CREDIT_CARD_BALANCE,
    "installments_payments": RAW_INSTALLMENTS_PAYMENTS,
    "POS_CASH_balance": RAW_POS_CASH_BALANCE,
    "previous_application": RAW_PREVIOUS_APPLICATION
}

TABLE_PK = {
    "application_train": "SK_ID_CURR",
    "bureau": "SK_ID_BUREAU",
    "bureau_balance": "SK_ID_BUREAU",
    "credit_card_balance": "SK_ID_PREV",
    "installments_payments": "SK_ID_PREV",
    "POS_CASH_balance": "SK_ID_PREV",
    "previous_application": "SK_ID_PREV"
}

def update_rows(updates: dict):

    if not updates:
        return
    
    if "application_train" in updates:

        df = pd.read_csv(RAW_APPLICATION_TRAIN).set_index("SK_ID_CURR")
        updates_df = pd.DataFrame(updates["application_train"]).set_index("SK_ID_CURR")

        new_idx = updates_df.index.difference(df.index)
        up_idx = updates_df.index.intersection(df.index)

        df = pd.concat([df, updates_df.loc[new_idx]])

        null_up_idx = up_idx[df.loc[up_idx, "TARGET"].isna()]
        df.loc[null_up_idx, updates_df.columns] = updates_df.loc[null_up_idx, updates_df.columns]

        no_target_idx = up_idx[df.loc[up_idx, "TARGET"].isna()]
        proc_ids = pd.read_parquet(PROCESSED_IDS)
        proc_preds = pd.read_parquet(PROCESSED_PREDS)
        proc_x = pd.read_parquet(PROCESSED_X)
        proc_mask = ~proc_ids["SK_ID_CURR"].isin(no_target_idx)
        proc_ids.loc[proc_mask].to_parquet(PROCESSED_IDS, index=False)
        proc_preds.loc[proc_mask].to_parquet(PROCESSED_PREDS, index=False)
        proc_x.loc[proc_mask].to_parquet(PROCESSED_X, index=False)

        df.reset_index().to_csv(RAW_APPLICATION_TRAIN, index=False)

    for table in updates:

        if table == "application_train": continue

        df = pd.read_csv(TABLE_PATHS[table]).set_index(TABLE_PK[table])
        updates_df = pd.DataFrame(updates[table]).set_index(TABLE_PK[table])

        new_idx = updates_df.index.difference(df.index)

        df = pd.concat([df, updates_df[new_idx]])

        df.reset_index().to_csv(TABLE_PATHS[table], index=False)


    state = load_state()
    state["transform_required"] = True
    save_state(state)