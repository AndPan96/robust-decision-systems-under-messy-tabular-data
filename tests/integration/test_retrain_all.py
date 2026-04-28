from unittest.mock import patch
from src.pipelines.retrain_all import retrain_all_pipeline
from src.models.linear_model import LinearModel
from src.models.mlp import MLP
from src.models.prior_model import PriorModel

def test_retrain_all_pipeline(set_environment, reset_environment):

    try:
        set_environment()

        test_registry = {
            "PriorModel": {
                "class": PriorModel,
                "arch_params": {"num_classes": 2},
                "train_params": {"lr": "NA", "batch_size": "NA", "steps": "NA"}
            },
            "LinearModel": {
                "class": LinearModel,
                "arch_params": {"num_classes": 2},
                "train_params": {"lr": .01, "batch_size": 256, "steps": 2}
            },
            "MLP": {
                "class": MLP,
                "arch_params": {"num_classes": 2},
                "train_params": {"lr": .01, "batch_size": 256, "steps": 2}
            }
        }

        with patch("src.actions.train_model.MODEL_REGISTRY", test_registry):
            retrain_all_pipeline()

    finally:
        reset_environment()