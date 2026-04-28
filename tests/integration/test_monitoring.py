import pytest
from unittest.mock import patch
from src.pipelines.monitoring import monitoring_pipeline
from src.utils.reports import create_new_report_folder, REPORT_MONITORING_DIR
from src.config.registry import deploy_model
import pandas as pd
from src.models.linear_model import LinearModel
from sklearn.compose import ColumnTransformer

def test_monitoring_pipeline(set_environment, reset_environment):

    try:
        set_environment()

        report_path, plots_path = create_new_report_folder(REPORT_MONITORING_DIR)
        metrics_path = report_path / "metrics.parquet"
        pd.DataFrame(columns=["timestamp", "accuracy", "n_samples"]).to_parquet(metrics_path)
        model = LinearModel(1, 2)  
        preprocessor = ColumnTransformer([])  

        deploy_model(
            model,
            preprocessor,
            {
                "model_class_name": "LinearModel",
                "metrics": {},
                "input_dim": 1,
                "report_path": report_path,
                "plots_path": plots_path
            }
        )

        with patch("src.pipelines.monitoring.monitor_model") as mock_monitor_model, \
            patch("src.pipelines.monitoring.load_state") as mock_load_state, \
            patch("src.pipelines.monitoring.save_state") as mock_save_state:

            mock_monitor_model.return_value = None

            monitoring_pipeline()

            assert mock_monitor_model.call_count == 1
            assert mock_load_state.call_count == 0
            assert mock_save_state.call_count == 0

        with patch("src.pipelines.monitoring.monitor_model") as mock_monitor_model, \
            patch("src.pipelines.monitoring.load_state") as mock_load_state, \
            patch("src.pipelines.monitoring.save_state") as mock_save_state:

            mock_monitor_model.return_value = {
                "accuracy": .9,
                "retrain": False,
                "n_samples": 100
            }

            monitoring_pipeline()

            assert mock_monitor_model.call_count == 1
            assert mock_load_state.call_count == 0
            assert mock_save_state.call_count == 0

        with patch("src.pipelines.monitoring.monitor_model") as mock_monitor_model, \
            patch("src.pipelines.monitoring.load_state") as mock_load_state, \
            patch("src.pipelines.monitoring.save_state") as mock_save_state:

            mock_monitor_model.return_value = {
                "accuracy": .9,
                "retrain": True,
                "n_samples": 100
            }

            mock_load_state.return_value = {"retrain_required": False}

            monitoring_pipeline()

            assert mock_monitor_model.call_count == 1
            assert mock_load_state.call_count == 1
            assert mock_save_state.call_count == 1

            saved_state = mock_save_state.call_args[0][0]
            assert saved_state["retrain_required"] == True

    finally:
        reset_environment()