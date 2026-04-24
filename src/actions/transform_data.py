from src.config.state import load_state, save_state
from src.config.paths import DATASET_PATH, RAW_APPLICATION_TRAIN, RAW_BUREAU, RAW_BUREAU_BALANCE, \
    RAW_CREDIT_CARD_BALANCE, RAW_INSTALLMENTS_PAYMENTS, RAW_POS_CASH_BALANCE, RAW_PREVIOUS_APPLICATION
import pandas as pd
from typing import cast

FEATURE_SCHEMA = {
    "id": "SK_ID_CURR",
    "target": "TARGET",

    "mean": [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",

        "B_ENDDATE_MEAN", "B_ENDDATE_STD",
        "B_CREDIT_MEAN", "B_CREDIT_STD",
        "B_DEBT_MEAN", "B_DEBT_STD",

        "BB_MB_MEAN", "BB_MB_STD",

        "PA_AMT_ANNUITY_MEAN", "PA_AMT_ANNUITY_STD",
        "PA_AMT_APPLICATION_MEAN", "PA_AMT_APPLICATION_STD",
        "PA_AMT_CREDIT_MEAN", "PA_AMT_CREDIT_STD",
        "PA_AMT_DOWN_PAYMENT_MEAN", "PA_AMT_DOWN_PAYMENT_STD",
        "PA_NAME_YIELD_MEAN",

        "PCB_DPD_MEAN", "PCB_DPD_DEF_MEAN",

        "IP_DELAY_MEAN", "IP_DELAY_STD",
        "IP_AMT_DIFF_MEAN", "IP_AMT_DIFF_STD",

        "CCB_AMT_BALANCE_MEAN", "CCB_AMT_BALANCE_STD",
        "CCB_SK_DPD_MEAN", "CCB_SK_DPD_STD",
        "CCB_SK_DPD_DEF_MEAN", "CCB_SK_DPD_DEF_STD",
    ],

    "zero": [
        "CNT_CHILDREN",
        "B_COUNT",
        "BB_COUNT",
        "PA_COUNT",
    ],

    "binary": [
        "FLAG_MOBIL", "FLAG_EMAIL", "LIVE_CITY_NOT_WORK_CITY",
        "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
    ]
}

def transform_data():

    state = load_state()
    if state["transform_required"]:

        application_train = pd.read_csv(RAW_APPLICATION_TRAIN)
        bureau = pd.read_csv(RAW_BUREAU)
        bureau_balance = pd.read_csv(RAW_BUREAU_BALANCE)
        credit_card_balance = pd.read_csv(RAW_CREDIT_CARD_BALANCE)
        installments_payments = pd.read_csv(RAW_INSTALLMENTS_PAYMENTS)
        pos_cash_balance = pd.read_csv(RAW_POS_CASH_BALANCE)
        previous_application = pd.read_csv(RAW_PREVIOUS_APPLICATION)


        df = application_train[["SK_ID_CURR", "TARGET", 
                               "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                               "FLAG_MOBIL", "FLAG_EMAIL", "LIVE_CITY_NOT_WORK_CITY",
                               "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5",
                               "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9",
                               "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
                               "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17",
                               "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"]]


        bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg({"MONTHS_BALANCE": ["mean","std"]})
        bb_agg.columns = ["BB_MB_MEAN", "BB_MB_STD"]
        bb_agg = bb_agg.reset_index()
        bb_count = bureau_balance.groupby("SK_ID_BUREAU").size()
        bb_count.name = "BB_COUNT"
        bb_count = bb_count.reset_index()
        bb_agg = bb_agg.merge(bb_count, on="SK_ID_BUREAU")
        bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

        b_agg = bureau.groupby("SK_ID_CURR").agg({
            "DAYS_ENDDATE_FACT": ["mean","std"],
            "CNT_CREDIT_PROLONG": ["mean","std"],
            "AMT_CREDIT_SUM": ["mean","std"],
            "AMT_CREDIT_SUM_DEBT": ["mean","std"],
            "BB_MB_MEAN": ["mean"],
            "BB_MB_STD": ["mean"],
            "BB_COUNT": ["mean"]
        })
        b_agg.columns = ["B_ENDDATE_MEAN", "B_ENDDATE_STD", "B_PROLONG_MEAN", "B_PROLONG_STD",
                         "B_CREDIT_MEAN", "B_CREDIT_STD", "B_DEBT_MEAN", "B_DEBT_STD",
                         "BB_MB_MEAN", "BB_MB_STD", "BB_COUNT"]
        b_agg = b_agg.reset_index()
        b_count = bureau["BB_COUNT"].notna().groupby(bureau["SK_ID_CURR"]).sum()
        b_count.name = "B_COUNT"
        b_count = b_count.reset_index()
        b_agg = b_agg.merge(b_count, on="SK_ID_CURR")


        pcb = pos_cash_balance.groupby("SK_ID_PREV").agg({
            "SK_DPD": ["mean"],
            "SK_DPD_DEF": ["mean"]
        })
        pcb.columns = ["PCB_DPD_MEAN", "PCB_DPD_DEF_MEAN"]
        pcb = pcb.reset_index()

        installments_payments["DAYS_INSTALLMENT_DELAY"] = installments_payments["DAYS_ENTRY_PAYMENT"] - installments_payments["DAYS_INSTALMENT"]
        installments_payments["AMT_PAYMENT_DIFF"] = installments_payments["AMT_PAYMENT"] - installments_payments["AMT_INSTALMENT"]
        ip = installments_payments.groupby("SK_ID_PREV").agg({
            "DAYS_INSTALLMENT_DELAY": ["mean","std"],
            "AMT_PAYMENT_DIFF": ["mean","std"]
        })
        ip.columns = ["IP_DELAY_MEAN", "IP_DELAY_STD", "IP_AMT_DIFF_MEAN", "IP_AMT_DIFF_STD"]
        ip = ip.reset_index()

        ccb = credit_card_balance.groupby("SK_ID_PREV").agg({
            "AMT_BALANCE": ["mean","std"],
            "SK_DPD": ["mean","std"],
            "SK_DPD_DEF": ["mean","std"]
        })
        ccb.columns = ["CCB_AMT_BALANCE_MEAN", "CCB_AMT_BALANCE_STD", "CCB_SK_DPD_MEAN", 
                       "CCB_SK_DPD_STD", "CCB_SK_DPD_DEF_MEAN", "CCB_SK_DPD_DEF_STD"]
        ccb = ccb.reset_index()

        previous_application = previous_application.merge(pcb, on="SK_ID_PREV", how="left")
        previous_application = previous_application.merge(ip, on="SK_ID_PREV", how="left")
        previous_application = previous_application.merge(ccb, on="SK_ID_PREV", how="left")
        name_yield_map = {
            'low_action': 0,
            'low_normal': 0,
            'middle': 1,
            'high': 2,
            'XNA': None
        }
        previous_application["NAME_YIELD"] = previous_application["NAME_YIELD_GROUP"].map(name_yield_map)
        papp = previous_application.groupby("SK_ID_CURR").agg({
            "AMT_ANNUITY": ["mean","std"],
            "AMT_APPLICATION": ["mean","std"],
            "AMT_CREDIT": ["mean","std"],
            "AMT_DOWN_PAYMENT": ["mean","std"],
            "NAME_YIELD": ["mean"],
            "PCB_DPD_MEAN": ["mean"],
            "PCB_DPD_DEF_MEAN": ["mean"],

            "IP_DELAY_MEAN": ["mean"],
            "IP_DELAY_STD": ["mean"],
            "IP_AMT_DIFF_MEAN": ["mean"],
            "IP_AMT_DIFF_STD": ["mean"],

            "CCB_AMT_BALANCE_MEAN": ["mean"],
            "CCB_AMT_BALANCE_STD": ["mean"],
            "CCB_SK_DPD_MEAN": ["mean"],
            "CCB_SK_DPD_STD": ["mean"],
            "CCB_SK_DPD_DEF_MEAN": ["mean"],
            "CCB_SK_DPD_DEF_STD": ["mean"]
        })
        papp.columns = ["PA_AMT_ANNUITY_MEAN", "PA_AMT_ANNUITY_STD", "PA_AMT_APPLICATION_MEAN", "PA_AMT_APPLICATION_STD",
            "PA_AMT_CREDIT_MEAN", "PA_AMT_CREDIT_STD", "PA_AMT_DOWN_PAYMENT_MEAN", "PA_AMT_DOWN_PAYMENT_STD",
            "PA_NAME_YIELD_MEAN", "PCB_DPD_MEAN", "PCB_DPD_DEF_MEAN", "IP_DELAY_MEAN", "IP_DELAY_STD",
            "IP_AMT_DIFF_MEAN", "IP_AMT_DIFF_STD", "CCB_AMT_BALANCE_MEAN", "CCB_AMT_BALANCE_STD",
            "CCB_SK_DPD_MEAN", "CCB_SK_DPD_STD", "CCB_SK_DPD_DEF_MEAN", "CCB_SK_DPD_DEF_STD"]
        papp = papp.reset_index()
        papp_count = previous_application.groupby("SK_ID_CURR").size().rename("PA_COUNT").reset_index()
        papp = papp.merge(papp_count, on="SK_ID_CURR", how="left")


        df = df.merge(b_agg, on="SK_ID_CURR", how="left")
        df = df.merge(papp, on="SK_ID_CURR", how="left")

        cols = (
            [FEATURE_SCHEMA["id"], FEATURE_SCHEMA["target"]] +
            FEATURE_SCHEMA["mean"] +
            FEATURE_SCHEMA["zero"] +
            FEATURE_SCHEMA["binary"]
        )

        df = cast(pd.DataFrame, df[cols])
        df.to_csv(DATASET_PATH, index=False)
        state["transform_required"] = False
        save_state(state)