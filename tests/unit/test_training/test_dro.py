import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.dro import dro_train_epoch

@pytest.fixture
def fake_model():
    return torch.nn.Linear(5, 3)

@pytest.fixture
def fake_loader():
    loaders =[]

    for _ in range(3):
        X = torch.randn(8, 5)
        y = torch.randint(0,1, (8,))
        df = TensorDataset(X, y)
        loader = DataLoader(df, batch_size=4)
        loaders.append(loader)

    return loaders

@pytest.fixture
def fake_optim(fake_model: torch.nn.Module):
    return torch.optim.SGD(fake_model.parameters(), lr=.1)

def test_dro(fake_model, fake_loader, fake_optim):

    avg_loss = dro_train_epoch(fake_model, fake_loader, fake_optim, "cpu")

    assert isinstance(avg_loss, float)
    assert avg_loss > 0