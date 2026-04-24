import pytest
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import patch
from src.actions.predict import predict
from src.config.paths import MONITORING_IDS, MONITORING_X
from src.config.registry import DEVICE
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

@pytest.fixture
def fake_load():
    X = pd.DataFrame({
          "F1": [.1, .2],
          "F2": [1, 2]
    })
    ids = pd.Series([1, 2])

    return ids, X

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)

@pytest.fixture
def fake_preprocessor():
    prep = ColumnTransformer(transformers=[("identiy", FunctionTransformer(),["F1", "F2"])])
    prep.set_output(transform="pandas")
    prep.fit(pd.DataFrame({"F1": [0], "F2": [0]}))
    return prep

def test_predict(fake_load, fake_preprocessor):

    try:

        ids: pd.Series
        X: pd.DataFrame
        ids, X = fake_load

        ids.to_frame("ID").to_parquet(MONITORING_IDS)
        X.to_parquet(MONITORING_X)

        fake_model = DummyModel().to(DEVICE)

        with patch("src.actions.predict.load_current_model") as mock_load_current_model, \
            patch("src.actions.predict.append_parquet") as mock_append_parquet:

            mock_load_current_model.return_value = (fake_model, fake_preprocessor, None)

            preds = predict()

            assert isinstance(preds, pd.DataFrame)
            assert list(preds.columns) == ["TARGET_PRED"]
            assert preds.shape[0] == len(X)

            assert mock_append_parquet.call_count == 3

    finally:
        MONITORING_IDS.unlink(missing_ok=True)
        MONITORING_X.unlink(missing_ok=True)
        