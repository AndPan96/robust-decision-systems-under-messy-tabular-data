from pathlib import Path
from src.models.prior_model import PriorModel
from src.models.linear_model import LinearModel
from src.models.mlp import MLP
import json
import torch
import joblib
from sklearn.compose import ColumnTransformer

MODEL_REGISTRY = {
    "PriorModel": {
            "class": PriorModel,
            "arch_params": {
                "num_classes": 2
            },
            "train_params": {
                "lr" : "NA",
                "batch_size" : "NA",
                "steps": "NA"
            }
        },
    "LinearModel": {
            "class": LinearModel,
            "arch_params": {
                "num_classes": 2
            },
            "train_params": {
                "lr" : .001,
                "batch_size" : 256,
                "steps": 35
            }
        },
    "MLP": {
        "class": MLP,
            "arch_params": {
                "num_classes": 2
            },
            "train_params": {
                "lr" : .001,
                "batch_size" : 256,
                "steps": 35
            }
        }
}

MODEL_PATH = Path("saved_models/current_model.pt")
PREPROCESSOR_PATH = Path("saved_models/current_preprocessor.pkl")
MODEL_META = Path("saved_models/current_model.json")
DEVICE = "cuda"

def load_current_model():
    
    with open(MODEL_META) as f:
        meta = json.load(f)

    meta["report_path"] = Path(meta["report_path"])
    meta["plots_path"] = Path(meta["plots_path"])

    model_class_name = meta["model_class_name"]
    model_input_dim = meta["input_dim"]
    model_class = MODEL_REGISTRY[model_class_name]["class"]
    model_class_arch = MODEL_REGISTRY[model_class_name]["arch_params"]
    
    if model_class == PriorModel:
        model : torch.nn.Module = model_class(**model_class_arch)
    else:
        model : torch.nn.Module = model_class(input_dim=model_input_dim,**model_class_arch)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    preprocessor: ColumnTransformer = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor, meta

def deploy_model(model : torch.nn.Module, preprocessor: ColumnTransformer, meta: dict):

    torch.save(model.state_dict(), MODEL_PATH)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    meta["report_path"] = str(meta["report_path"])
    meta["plots_path"] = str(meta["plots_path"])
    with open(MODEL_META, "w") as f:
        json.dump(meta, f, indent = 2)