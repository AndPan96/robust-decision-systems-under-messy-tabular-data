import pytest
from src.actions.train_model import train_model
import torch
import torch.nn as nn
import pandas as pd
from unittest.mock import patch

@pytest.fixture
def fake_load():
    X = pd.DataFrame({
          "F1": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
          "F2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    return X, y

class DummyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.layer(x)

def test_train_model(fake_load):

    X, y = fake_load
    mysteps = 2

    with patch("src.actions.train_model.dro_train_epoch") as mock_dro_train_epoch, \
        patch("src.actions.train_model.generate_inner_fold_loaders") as mock_generate_inner_fold_loaders, \
        patch("src.actions.train_model.generate_outer_folds") as mock_generate_outer_folds, \
        patch("src.actions.train_model.preprocess_fit") as mock_preprocess_fit, \
        patch("src.actions.train_model.MODEL_REGISTRY",
        {
            "DummyModel": {
                "class": DummyModel,
                "arch_params": {},
                "train_params": {
                    "lr": .01,
                    "batch_size": 2,
                    "steps": mysteps
                },
                "input_dim": 2
            }
        }):
    
        X_tensor = torch.tensor(X.values, dtype=torch.float)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        X_val = pd.DataFrame(torch.randn(4, 2).numpy())
        y_val = pd.Series([1, 1, 0, 0])
        imp_col_size = 2

        mock_generate_outer_folds.return_value = ([(X_tensor, y_tensor, X_val, y_val),
                                                  (X_tensor, y_tensor, X_val, y_val),
                                                  (X_tensor, y_tensor, X_val, y_val),
                                                  (X_tensor, y_tensor, X_val, y_val),
                                                  (X_tensor, y_tensor, X_val, y_val)], imp_col_size)

        mock_generate_inner_fold_loaders.return_value = []

        mock_dro_train_epoch.return_value = .5

        mock_preprocess_fit.return_value = (X, None)

        mysplits = 5
        result = train_model(DummyModel, X, y, "TBD", mysplits, "cpu")

        assert isinstance(result, dict)
        assert "model" in result
        assert "metrics" in result
        assert "hyperparams" in result
        assert mock_dro_train_epoch.call_count == (mysplits + 1) * mysteps
        assert mock_generate_inner_fold_loaders.call_count == mysplits + 1
        assert mock_generate_outer_folds.call_count == 1