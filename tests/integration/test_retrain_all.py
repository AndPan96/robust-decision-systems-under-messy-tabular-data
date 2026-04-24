from unittest.mock import patch
from src.pipelines.retrain_all import retrain_all_pipeline

def test_retrain_all_pipeline(set_environment, reset_environment):

    try:
        set_environment()

        with patch("src.pipelines.retrain_all.train_split") as mock_train_split, \
            patch("src.pipelines.retrain_all.test_deploy_all") as mock_test_deploy_all:

            retrain_all_pipeline()

            assert mock_train_split.call_count == 1
            assert mock_test_deploy_all.call_count == 1

    finally:
        reset_environment()