import pytest
from src.actions.train_all import train_all
from unittest.mock import patch
import pandas as pd
import torch
import torch.nn as nn

@pytest.fixture
def fake_load():
    X = pd.DataFrame({
          "F1": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
          "F2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    return X, y

class DummyModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)

class DummyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)

def test_train_all(fake_load):

    X, y = fake_load

    hyperparams = {
        "lr": .01,
        "batch_size": 2,
        "steps": 2
    }

    FAKE_REGISTRY = {
            "DummyModel1": {
                "class": DummyModel1,
                "arch_params": {},
                "train_params": {
                    "lr": .01,
                    "batch_size": 2,
                    "steps": 2
                }
            },
            "DummyModel2": {
                "class": DummyModel2,
                "arch_params": {},
                "train_params": {
                    "lr": .01,
                    "batch_size": 2,
                    "steps": 2
                }
            }
    }

    with patch("src.actions.train_all.MODEL_REGISTRY", new=FAKE_REGISTRY), \
        patch("src.actions.train_all.train_model") as mock_train_model:

        mock_train_model.side_effect = [
            {"name": "model1"},
            {"name": "model2"}
        ]

        result = train_all(X, y, "TBD", 5, "cpu")

        assert len(result) == 2
        assert result[0]["name"] == "model1"
        assert result[1]["name"] == "model2"
        assert mock_train_model.call_count == len(FAKE_REGISTRY)


