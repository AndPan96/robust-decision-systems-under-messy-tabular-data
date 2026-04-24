import pytest
import pandas as pd
from src.actions.predict_split import predict_split
from unittest.mock import patch
from src.config.paths import MONITORING_IDS, MONITORING_X

def test_predict_split():

    try:
        with patch("pandas.DataFrame.to_parquet") as mock_to_parquet, \
            patch("src.actions.predict_split.pd.read_parquet") as mock_read_parquet, \
            patch("src.actions.predict_split.load_dataset") as mock_load_dataset:

            ids = pd.Series([1, 2, 3, 4])
            X = pd.DataFrame({"F1": [.1, .2, .3, .4], "F2": [1, 2, 3, 4]})
            y = pd.Series([1, None, None, 0])
            mock_load_dataset.return_value = (ids, X, y)
            mock_read_parquet.return_value = pd.DataFrame({"SK_ID_CURR": [2]})
        
            predict_split()

            assert mock_load_dataset.call_count == 1
            assert mock_to_parquet.call_count == 2
            
    finally:
        MONITORING_IDS.unlink(missing_ok=True)
        MONITORING_X.unlink(missing_ok=True)

