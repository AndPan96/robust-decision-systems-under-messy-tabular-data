from src.actions.train_split import train_split
from src.actions.train_model import train_model
from src.config.registry import load_current_model, deploy_model, DEVICE
from src.config.data_config import DEPLOY_THRESHOLD
from src.config.state import load_state, save_state
from typing import cast
import pandas as pd
import torch
from src.config.paths import TRAINING_X, TRAINING_Y, TEST_X, TEST_Y

def test_deploy_model():

    X_train = pd.read_parquet(TRAINING_X)
    y_train = pd.read_parquet(TRAINING_Y)["TARGET"]

    X_test = pd.read_parquet(TEST_X)
    y_test = pd.read_parquet(TEST_Y)["TARGET"]

    current_model, preprocessor, _ = load_current_model()
    X_test_imp = cast(pd.DataFrame, preprocessor.transform(X_test))

    new_model = train_model(type(current_model), X_train, y_train, "TBD", 5, DEVICE)

    model: torch.nn.Module = new_model["model"]
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_imp.values, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_test.values, dtype=torch.long).to(DEVICE)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1)
        test_accuracy = (preds == y_tensor).float().mean().item()

    state = load_state()

    if test_accuracy > DEPLOY_THRESHOLD:
        meta = {
            "name": new_model["name"],
            "class_model_name": new_model["class_model_name"],
            "metrics": new_model["metrics"],
            "input_dim": new_model["input_dim"]
        }
        deploy_model(new_model["model"], preprocessor, meta)
        state["retrain_required"] = False
        print("Same model retraining succeded: deployed.")
    else:
        state["retrain_single"] = True
        
        print("Same model retraining failed.")

    save_state(state)