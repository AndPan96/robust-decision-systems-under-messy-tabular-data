import pytest
import pandas as pd
from torch.utils.data import DataLoader 
from src.training.dro_folds import generate_inner_fold_loaders

@pytest.fixture
def fake_load():
    X = pd.DataFrame({
          "F1": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
          "F2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    return X, y

def test_dro_folds(fake_load):

    X, y = fake_load

    loaders = generate_inner_fold_loaders(X, y, 2, 256)

    assert isinstance(loaders, list)
    assert len(loaders) == 2
    assert isinstance(loaders[0], DataLoader)