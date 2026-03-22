from src.actions.train_all import train_all
from src.config.registry import deploy_model, DEVICE
from src.config.data_config import DEPLOY_THRESHOLD
from src.config.state import load_state, save_state
import pandas as pd
import torch
from src.config.paths import TRAINING_X, TRAINING_Y, TEST_X, TEST_Y

def test_deploy_all():

    X_train = pd.read_parquet(TRAINING_X)
    y_train = pd.read_parquet(TRAINING_Y)["TARGET"]

    X_test = pd.read_parquet(TEST_X)
    y_test = pd.read_parquet(TEST_Y)["TARGET"]

    models = train_all(X_train, y_train, "TBD", 5, DEVICE)
    best_model = max(models, key= lambda m : m["metrics"]["accuracy_mean"])

    model: torch.nn.Module = best_model["model"]
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_test.values, dtype=torch.long).to(DEVICE)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1)
        test_accuracy = (preds == y_tensor).float().mean().item()

    state = load_state()

    if test_accuracy > DEPLOY_THRESHOLD:
        meta = {
            "name": best_model["name"],
            "class_model_name": best_model["class_model_name"],
            "metrics": best_model["metrics"]
        }
        deploy_model(best_model["model"], meta)
        state["retrain_required"] = False
        state["retrain_single"] = False
        print("All models retraining succeded: deployed.")
    else:
        state["retrain_all"] = True
        print("All models retraining failed.")

    save_state(state)