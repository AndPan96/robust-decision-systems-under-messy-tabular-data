import pytest
import pandas as pd
from src.actions.predict_split import predict_split
from unittest.mock import patch

def test_predict_split():

    with patch("src.actions.predict_split.save_checkpoint") as mock_save_checkpoint, \
        patch("pandas.DataFrame.to_parquet") as mock_to_parquet, \
        patch("src.actions.predict_split.load_checkpoint") as mock_load_checkpoint, \
        patch("src.actions.predict_split.load_dataset") as mock_load_dataset:

        ids = pd.Series([1, 2, 3, 4])
        X = pd.DataFrame({"F1": [.1, .2, .3, .4], "F2": [1, 2, 3, 4]})
        y = pd.Series([1, None, None, 0])
        mock_load_dataset.return_value = (ids, X, y)

        mock_load_checkpoint.return_value = 1
    
        predict_split()

        assert mock_load_dataset.call_count == 1
        assert mock_load_checkpoint.call_count == 1
        assert mock_to_parquet.call_count == 2
        assert mock_save_checkpoint.call_count == 1