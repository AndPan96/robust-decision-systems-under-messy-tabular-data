from unittest.mock import patch
from src.pipelines.retrain_single import retrain_single_pipeline

def test_retrain_single_pipeline():

    with patch("src.pipelines.retrain_single.train_split") as mock_train_split, \
        patch("src.pipelines.retrain_single.test_deploy_model") as mock_test_deploy_model:

        retrain_single_pipeline()

        assert mock_train_split.call_count == 1
        assert mock_test_deploy_model.call_count == 1