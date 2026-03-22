import pytest
import pandas as pd
from src.data.split import initial_split

@pytest.fixture
def fake_load():

     X = pd.DataFrame({
          "F1": [.1, .2, .3, .4],
          "F2": [1, 2, 3, 4]
     })
     y = pd.Series([1, 0, 1, 0])

     return X, y

def test_init_split(fake_load):

    X, y = fake_load

    X_train, y_train, X_test, y_test = initial_split(X, y, 1, 1)

    assert X_train.shape[0] == 1
    assert y_train.shape[0] == 1
    assert X_test.shape[0] == 1
    assert y_test.shape[0] == 1
    assert y_train.isna().sum() == 0
    assert y_test.isna().sum() == 0