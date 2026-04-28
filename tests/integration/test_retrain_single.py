from unittest.mock import patch
from src.pipelines.retrain_single import retrain_single_pipeline
from src.models.linear_model import LinearModel
from src.config.registry import deploy_model
from src.actions.transform_data import FEATURE_SCHEMA
from sklearn.compose import ColumnTransformer

def test_retrain_single_pipeline(set_environment, reset_environment):

    try:
        set_environment()

        test_registry = {
            "LinearModel": {
                "class": LinearModel,
                "arch_params": {"num_classes": 2},
                "train_params": {"lr": .01, "batch_size": 256, "steps": 2}
            }
        }

        with patch("src.actions.train_model.MODEL_REGISTRY", test_registry):

            feature_cols = (
                FEATURE_SCHEMA["mean"] +
                FEATURE_SCHEMA["zero"] +
                FEATURE_SCHEMA["binary"]
            )
            model = LinearModel(len(feature_cols), 2)  
            preprocessor = ColumnTransformer([])  
            deploy_model(
                model,
                preprocessor,
                {
                    "model_class_name": "LinearModel",
                    "metrics": {},
                    "input_dim": len(feature_cols),
                    "report_path": "",
                    "plots_path": ""
                }
            )

            retrain_single_pipeline()

    finally:
        reset_environment()