import pytest
import torch
from src.models.prior_model import PriorModel
from src.models.linear_model import LinearModel
from src.models.mlp import MLP

@pytest.fixture
def fake_data():
    X = torch.tensor([[.1,1],
                      [.2,2],
                      [.3,3],
                      [.4,4]], dtype=torch.float32)
    return X

def test_prior_model(fake_data):
    model = PriorModel(2)
    preds = model(fake_data)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape[0] == fake_data.shape[0]
    assert preds.shape[1] == 2

def test_linear_model(fake_data):
    model = LinearModel(2, 2)
    preds = model(fake_data)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape[0] == fake_data.shape[0]
    assert preds.shape[1] == 2

def test_mlp(fake_data):
    model = MLP(2, 2)
    preds = model(fake_data)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape[0] == fake_data.shape[0]
    assert preds.shape[1] == 2


