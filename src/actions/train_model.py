import torch
from src.training.dro import dro_train_epoch
from src.training.dro_folds import generate_inner_fold_loaders
from src.training.cross_validation import generate_outer_folds
from src.config.registry import MODEL_REGISTRY
from src.actions.transform_data import FEATURE_SCHEMA
from src.models.prior_model import PriorModel
from typing import cast
import numpy as np

def build_model_name(model_class, data_version, splits, hyperparams):

    name = (
        f"{model_class.__name__}"
        f"_dv{data_version}"
        f"_splits{splits}"
        f"_lr{hyperparams['lr']}"
        f"_bs{hyperparams['batch_size']}"
        f"_steps{hyperparams['steps']}"
    )

    return name

def train_model(model_class, X, y, data_version, splits, device):

    feature_cols = (
        FEATURE_SCHEMA["mean"] +
        FEATURE_SCHEMA["zero"] +
        FEATURE_SCHEMA["binary"]
    )
    model_class_arch = MODEL_REGISTRY[model_class.__name__]["arch_params"]  
    hyperparams = MODEL_REGISTRY[model_class.__name__]["train_params"]
    criterion = torch.nn.CrossEntropyLoss()
    metrics_store = {
        "loss": [],
        "accuracy": []
    }

    kf = generate_outer_folds(X, y, splits)

    for X_train, y_train, X_val, y_val in kf:

        model: torch.nn.Module
        model = model_class(input_dim=len(feature_cols),**model_class_arch).to(device)

        if isinstance(model, PriorModel):
            model.fit(torch.tensor(y_train.values, dtype=torch.long))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr = hyperparams['lr'])
            fold_loaders = generate_inner_fold_loaders(X_train, y_train, batch_size = hyperparams['batch_size'])

            for _ in range(hyperparams['steps']):
                dro_train_epoch(model, fold_loaders, optimizer, device)

        model.eval()
        with torch.no_grad():
            
            X_val = X_val.to(device)
            y_val: torch.Tensor = y_val.to(device)

            logits = model(X_val)

            loss = cast(torch.Tensor, criterion(logits, y_val)).item()

            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_val).float().mean().item()

        metrics_store["loss"].append(loss)
        metrics_store["accuracy"].append(accuracy)

    model: torch.nn.Module
    model = model_class(input_dim=len(feature_cols),**model_class_arch).to(device)

    if isinstance(model, PriorModel):
        model.fit(torch.tensor(y.values, dtype=torch.long))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = hyperparams['lr'])
        fold_loaders = generate_inner_fold_loaders(X, y, batch_size = hyperparams['batch_size'])

        for _ in range(hyperparams['steps']):
            dro_train_epoch(model, fold_loaders, optimizer, device)

    metrics = {}

    for metrics_name, values in metrics_store.items():

        metrics[f"{metrics_name}_mean"] = float(np.mean(values))
        metrics[f"{metrics_name}_std"] = float(np.std(values))

    model_name = build_model_name(model_class, data_version, splits, hyperparams)

    return {
        "name": model_name,
        "model_class_name": model_class.__name__,
        "model": model,
        "metrics": metrics,
        "input_dim": len(feature_cols),
        "hyperparams": hyperparams
    }