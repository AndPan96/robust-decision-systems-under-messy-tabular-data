import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def generate_inner_fold_loaders(X: pd.DataFrame, y: pd.Series, n_splits=3, batch_size = 256):

    kf = KFold(n_splits=n_splits, shuffle=False)

    loaders = []

    for idx, _ in kf.split(X):

        X_fold = torch.tensor(X.iloc[idx].values, dtype=torch.float32)
        y_fold = torch.tensor(y.iloc[idx].values, dtype=torch.long)

        dataset = TensorDataset(X_fold, y_fold)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loaders.append(loader)

    return loaders