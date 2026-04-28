import pytest
import pandas as pd
from unittest.mock import patch
from src.training.cross_validation import generate_outer_folds
from sklearn.preprocessing import FunctionTransformer

@pytest.fixture
def fake_load():
    X = pd.DataFrame({
          "F1": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
          "F2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    return X, y

def fake_preprocess_fit(X):
        return X, FunctionTransformer()

def test_cross_validation(fake_load):

    X, y = fake_load

    

    with patch("src.training.cross_validation.preprocess_fit", side_effect=fake_preprocess_fit):
        folds, imp_col_len = generate_outer_folds(X, y, 5)

    assert isinstance(folds, list)
    assert len(folds) == 5
    
    X_train, y_train, X_val, y_val = folds[0]

    assert len(X_train) + len(X_val) == 10
    assert len(y_train) + len(y_val) == 10
    assert len(X_train) == len(y_train)