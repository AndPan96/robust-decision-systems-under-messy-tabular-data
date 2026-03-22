import pandas as pd
from typing import cast

def initial_split(X, y, train_window, test_window):

    mask = cast(pd.Series, y).notna()
    X = X[mask]
    y = y[mask]

    if len(X) < train_window + test_window:
        raise ValueError("Dataset too small for requested windows")
    
    split_idx = len(X) - test_window

    X_train = X.iloc[split_idx - train_window:split_idx]
    y_train = y.iloc[split_idx - train_window:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test