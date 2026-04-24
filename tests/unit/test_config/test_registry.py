import pytest
import torch
from unittest.mock import patch
from src.config.registry import load_current_model, deploy_model, MODEL_PATH, PREPROCESSOR_PATH, MODEL_META 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

@pytest.fixture
def fake_model():
    return torch.nn.Identity()

@pytest.fixture
def fake_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("identity", FunctionTransformer(), [0])
        ]
    )

@pytest.fixture
def fake_meta():
    return {
        "model_class_name": "dummy",
        "metrics": {
            "accuracy": .5
        },
        "input_dim": 1
    }

def test_registry(fake_model, fake_preprocessor, fake_meta):

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
        
        deploy_model(fake_model, fake_preprocessor, fake_meta)

        model, preprocessor, metrics = load_current_model()

        assert isinstance(model, torch.nn.Module)
        assert isinstance(preprocessor, ColumnTransformer)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert metrics.get("accuracy", .0) == .5

        MODEL_PATH.unlink(missing_ok=True)
        PREPROCESSOR_PATH.unlink(missing_ok=True)
        MODEL_META.unlink(missing_ok=True)

