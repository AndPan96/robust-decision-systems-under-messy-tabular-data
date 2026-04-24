import pandas as pd
from pathlib import Path

import src.actions.transform_data as mod
from src.actions.transform_data import FEATURE_SCHEMA


def test_transform_data_full_deterministic(tmp_path):

    app = pd.DataFrame({
        "SK_ID_CURR": [1],
        "TARGET": [0],
        "CNT_CHILDREN": [1],
        "AMT_INCOME_TOTAL": [1000],
        "AMT_CREDIT": [5000],
        "AMT_ANNUITY": [100],

        "FLAG_MOBIL": [1],
        "FLAG_EMAIL": [0],
        "LIVE_CITY_NOT_WORK_CITY": [0],

        "FLAG_DOCUMENT_2": [0],
        "FLAG_DOCUMENT_3": [1],
        "FLAG_DOCUMENT_4": [0],
        "FLAG_DOCUMENT_5": [0],
        "FLAG_DOCUMENT_6": [0],
        "FLAG_DOCUMENT_7": [0],
        "FLAG_DOCUMENT_8": [0],
        "FLAG_DOCUMENT_9": [0],
        "FLAG_DOCUMENT_10": [0],
        "FLAG_DOCUMENT_11": [0],
        "FLAG_DOCUMENT_12": [0],
        "FLAG_DOCUMENT_13": [0],
        "FLAG_DOCUMENT_14": [0],
        "FLAG_DOCUMENT_15": [0],
        "FLAG_DOCUMENT_16": [0],
        "FLAG_DOCUMENT_17": [0],
        "FLAG_DOCUMENT_18": [0],
        "FLAG_DOCUMENT_19": [0],
        "FLAG_DOCUMENT_20": [0],
        "FLAG_DOCUMENT_21": [0]
    })
    

    bureau = pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_BUREAU": [10],
        "DAYS_ENDDATE_FACT": [0],
        "CNT_CREDIT_PROLONG": [0],
        "AMT_CREDIT_SUM": [100],
        "AMT_CREDIT_SUM_DEBT": [50],
    })

    bb = pd.DataFrame({
        "SK_ID_BUREAU": [10],
        "MONTHS_BALANCE": [-1]
    })

    pcb = pd.DataFrame({
        "SK_ID_PREV": [100],
        "SK_DPD": [2],
        "SK_DPD_DEF": [1]
    })

    ip = pd.DataFrame({
        "SK_ID_PREV": [100],
        "DAYS_ENTRY_PAYMENT": [10],
        "DAYS_INSTALMENT": [5],
        "AMT_PAYMENT": [200],
        "AMT_INSTALMENT": [180],
    })

    ccb = pd.DataFrame({
        "SK_ID_PREV": [100],
        "AMT_BALANCE": [300],
        "SK_DPD": [2],
        "SK_DPD_DEF": [1]
    })

    pa = pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_PREV": [100],
        "AMT_ANNUITY": [100],
        "AMT_APPLICATION": [200],
        "AMT_CREDIT": [150],
        "AMT_DOWN_PAYMENT": [50],
        "NAME_YIELD_GROUP": ["middle"]
    })


    def write(df : pd.DataFrame, name):
        p = tmp_path / name
        df.to_csv(p, index=False)
        return p

    app_path = write(app, "app.csv")
    bureau_path = write(bureau, "bureau.csv")
    bb_path = write(bb, "bb.csv")
    pcb_path = write(pcb, "pcb.csv")
    ip_path = write(ip, "ip.csv")
    ccb_path = write(ccb, "ccb.csv")
    pa_path = write(pa, "pa.csv")
    out_path = tmp_path / "out.csv"


    mod.RAW_APPLICATION_TRAIN = app_path
    mod.RAW_BUREAU = bureau_path
    mod.RAW_BUREAU_BALANCE = bb_path
    mod.RAW_POS_CASH_BALANCE = pcb_path
    mod.RAW_INSTALLMENTS_PAYMENTS = ip_path
    mod.RAW_CREDIT_CARD_BALANCE = ccb_path
    mod.RAW_PREVIOUS_APPLICATION = pa_path
    mod.DATASET_PATH = out_path

    mod.save_state({"transform_required": True})


    mod.transform_data()
    df = pd.read_csv(out_path)

    expected_cols = (
        [FEATURE_SCHEMA["id"], FEATURE_SCHEMA["target"]] +
        FEATURE_SCHEMA["mean"] +
        FEATURE_SCHEMA["zero"] +
        FEATURE_SCHEMA["binary"]
    )


    assert list(df.columns) == expected_cols
    assert df.shape[0] == 1

    row = df.iloc[0]

    assert row["SK_ID_CURR"] == 1
    assert row["TARGET"] == 0
    assert row["CNT_CHILDREN"] == 1
    assert row["AMT_INCOME_TOTAL"] == 1000
    assert row["B_CREDIT_MEAN"] == 100
    assert row["B_DEBT_MEAN"] == 50
    assert row["BB_MB_MEAN"] == -1
    assert row["PCB_DPD_MEAN"] == 2
    assert row["IP_DELAY_MEAN"] == 5  # (10 - 5)
    assert row["IP_AMT_DIFF_MEAN"] == 20  # (200 - 180)
    assert row["CCB_AMT_BALANCE_MEAN"] == 300
    assert row["PA_AMT_CREDIT_MEAN"] == 150
    assert row["PA_COUNT"] == 1