import pytest
import pandas as pd
from src.actions.train_split import train_split
from unittest.mock import patch


def test_train_split():

    with patch("pandas.DataFrame.to_parquet") as mock_to_parquet, \
        patch("src.actions.train_split.initial_split") as mock_initial_split, \
        patch("src.actions.train_split.load_dataset") as mock_load_dataset:

        X = pd.DataFrame({"F1": [.1, .2, .3, .4], "F2": [1, 2, 3, 4]})
        y = pd.Series([1, 1, 0, 0])
        mock_load_dataset.return_value = (None, X, y)

        X_train = X.iloc[:2]
        y_train = y.iloc[:2]
        X_test = X.iloc[2:]
        y_test = y.iloc[2:]

        mock_initial_split.return_value = (X_train, y_train, X_test, y_test)

        train_split()

        assert mock_load_dataset.call_count == 1
        assert mock_initial_split.call_count == 1
        assert mock_to_parquet.call_count == 4
