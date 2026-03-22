import pytest
import torch
from unittest.mock import patch
from src.config.registry import load_current_model, deploy_model

@pytest.fixture
def fake_model():
    return torch.nn.Identity()

@pytest.fixture
def fake_meta():
    return {
        "model_class_name": "dummy",
        "metrics": {
            "accuracy": .5
        }
    }

def test_registry(fake_model, fake_meta):

    with patch(
        "src.config.registry.MODEL_REGISTRY",
        {
            "dummy": {
                "class": torch.nn.Identity,
                "arch_params": {},
                "train_params": {}
            }
        }
    ):
        
        deploy_model(fake_model, fake_meta)

        model, metrics = load_current_model()

        assert isinstance(model, torch.nn.Module)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert metrics.get("accuracy", .0) == .5