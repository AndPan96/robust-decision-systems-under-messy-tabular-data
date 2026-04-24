from sklearn.model_selection import KFold
import pandas as pd
from typing import cast
from src.actions.preprocess import preprocess_fit

def generate_outer_folds(X: pd.DataFrame, y: pd.Series, n_splits=5):

    kf = KFold(n_splits=n_splits, shuffle=False)

    folds = []

    for train_idx, val_idx in kf.split(X):

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        X_train_imp, preprocessor = preprocess_fit(X_train)
        X_train_imp = cast(pd.DataFrame, X_train_imp)
        X_val_imp = cast(pd.DataFrame, preprocessor.transform(X_val))

        folds.append((X_train_imp, y_train, X_val_imp, y_val))

    return folds