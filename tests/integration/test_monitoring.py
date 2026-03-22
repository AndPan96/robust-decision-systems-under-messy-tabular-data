import pytest
from unittest.mock import patch
from src.pipelines.monitoring import monitoring_pipeline

def test_monitoring_pipeline():

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
            "retrain": False
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
            "retrain": True
        }

        mock_load_state.return_value = {"retrain_required": False}

        monitoring_pipeline()

        assert mock_monitor_model.call_count == 1
        assert mock_load_state.call_count == 1
        assert mock_save_state.call_count == 1

        saved_state = mock_save_state.call_args[0][0]
        assert saved_state["retrain_required"] == True