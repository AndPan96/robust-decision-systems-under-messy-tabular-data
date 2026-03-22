from sklearn.model_selection import KFold
import pandas as pd

def generate_outer_folds(X: pd.DataFrame, y: pd.Series, n_splits=5):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds = []

    for train_idx, val_idx in kf.split(X):

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        folds.append((X_train, y_train, X_val, y_val))

    return folds