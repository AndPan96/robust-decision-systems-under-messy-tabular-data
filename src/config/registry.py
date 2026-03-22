from src.models.linear_model import LinearModel
from src.models.mlp import MLP
import json
import torch

MODEL_REGISTRY = {
    "LinearModel": {
            "class": LinearModel,
            "arch_params": {
                "input_dim": 2,
                "num_classes": 2
            },
            "train_params": {
                "lr" : .01,
                "batch_size" : 256
            }
        },
    "MLP": {
        "class": MLP,
            "arch_params": {
                "input_dim": 2,
                "num_classes": 2
            },
            "train_params": {
                "lr" : .01,
                "batch_size" : 256
            }
        }
}

MODEL_PATH = "saved_models/current_model.pt"
MODEL_META = "saved_models/current_model.json"
DEVICE = "cuda"

def load_current_model():
    
    with open(MODEL_META) as f:
        meta = json.load(f)

    model_class_name = meta["model_class_name"]
    model_class = MODEL_REGISTRY[model_class_name]["class"]
    model_class_arch = MODEL_REGISTRY[model_class_name]["arch_params"]
    model : torch.nn.Module = model_class(**model_class_arch)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    return model, meta["metrics"]

def deploy_model(model : torch.nn.Module, meta: dict):

    torch.save(model.state_dict(), MODEL_PATH)

    with open(MODEL_META, "w") as f:
        json.dump(meta, f, indent = 2)