from src.actions.train_all import train_all
from src.config.registry import deploy_model, DEVICE
from src.config.data_config import DEPLOY_THRESHOLD, CV_FOLDS
from src.config.state import load_state, save_state
from typing import cast
import pandas as pd
import torch
from src.actions.preprocess import preprocess_fit
from src.config.paths import TRAINING_X, TRAINING_Y, TEST_X, TEST_Y
from src.utils.email import send_email
from src.utils.reports import create_new_report_folder, REPORT_MONITORING_DIR

def test_deploy_all():

    X_train = pd.read_parquet(TRAINING_X)
    y_train = pd.read_parquet(TRAINING_Y)["TARGET"]

    X_test = pd.read_parquet(TEST_X)
    y_test = pd.read_parquet(TEST_Y)["TARGET"]

    _, preprocessor = preprocess_fit(X_train)
    models = train_all(X_train, y_train, "TBD", CV_FOLDS, DEVICE)
    best_model = max(models, key= lambda m : m["metrics"]["accuracy_mean"])

    X_test_imp = cast(pd.DataFrame, preprocessor.transform(X_test))


    model: torch.nn.Module = best_model["model"]
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_imp.values, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_test.values, dtype=torch.long).to(DEVICE)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1)
        test_accuracy = (preds == y_tensor).float().mean().item()


    state = load_state()

    if test_accuracy > DEPLOY_THRESHOLD:
        report_path, plots_path = create_new_report_folder(REPORT_MONITORING_DIR)
        meta = {
            "name": best_model["name"],
            "model_class_name": best_model["model_class_name"],
            "metrics": best_model["metrics"],
            "input_dim": best_model["input_dim"],
            "report_path": report_path,
            "plots_path": plots_path
        }
        deploy_model(best_model["model"], preprocessor, meta)
        state["retrain_required"] = False
        state["retrain_single"] = False
        print("All models retraining succeded: deployed.")
    else:
        state["retrain_all"] = True
        send_email("Retrain FAILED", 
            f"Retrain Failed.\n Best Accuracy:{test_accuracy:.4f}\n Threshold:{DEPLOY_THRESHOLD:.4f}")
        print("All models retraining failed.")

    save_state(state)

    return models