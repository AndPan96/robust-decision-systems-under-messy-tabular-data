import pandas as pd
import numpy as np
from typing import cast
from src.actions.preprocess import preprocess_fit
from src.actions.transform_data import FEATURE_SCHEMA


def test_preprocess_fit():

    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [1000, 2000, np.nan],
        "AMT_CREDIT": [5000, np.nan, 7000],
        "AMT_ANNUITY": [100, 200, 300],

        "CNT_CHILDREN": [1, np.nan, 3],
        "B_COUNT": [np.nan, 2, 3],
        "BB_COUNT": [1, 2, np.nan],
        "PA_COUNT": [np.nan, np.nan, 1],

        "FLAG_MOBIL": [1, 0, np.nan],       # valid + nan
        "FLAG_EMAIL": [0, 1, 2],            # invalid → UNKNOWN
        "LIVE_CITY_NOT_WORK_CITY": [1, 1, 1],  # all valid
    })

    for col in FEATURE_SCHEMA["binary"]:
        if col not in df.columns:
            df[col] = 0

    for col in FEATURE_SCHEMA["mean"]:
        if col not in df.columns:
            df[col] = 1.0

    for col in FEATURE_SCHEMA["zero"]:
        if col not in df.columns:
            df[col] = 0

 
    X_imp, preprocessor = preprocess_fit(df)


    assert isinstance(X_imp, pd.DataFrame)
    assert X_imp.shape[0] == 3
    assert not X_imp.isna().any().any()  # no NaNs after preprocessing


    for col in FEATURE_SCHEMA["mean"] + FEATURE_SCHEMA["zero"]:
        if col in X_imp.columns:
            assert abs(X_imp[col].mean()) < 1e-6
            std = X_imp[col].std(ddof=0)
            if np.isclose(std, 0):
                assert True
            else: 
                assert abs(X_imp[col].std(ddof=0) - 1) < 1e-6


    expected_cat_cols = [
        f"{col}_{val}"
        for col in FEATURE_SCHEMA["binary"]
        for val in ["0", "1", "UNKNOWN"]
    ]

    for col in expected_cat_cols:
        assert col in X_imp.columns


    assert X_imp.loc[0, "FLAG_MOBIL_1"] == 1.0
    assert X_imp.loc[1, "FLAG_MOBIL_0"] == 1.0
    assert X_imp.loc[2, "FLAG_MOBIL_UNKNOWN"] == 1.0  # NaN → UNKNOWN

    assert X_imp.loc[0, "FLAG_EMAIL_0"] == 1.0
    assert X_imp.loc[1, "FLAG_EMAIL_1"] == 1.0
    assert X_imp.loc[2, "FLAG_EMAIL_UNKNOWN"] == 1.0  # invalid (2) → UNKNOWN

    assert (X_imp["LIVE_CITY_NOT_WORK_CITY_1"] == 1.0).all()


    X_imp_2 = cast(pd.DataFrame, preprocessor.transform(df))
    pd.testing.assert_frame_equal(X_imp, X_imp_2)