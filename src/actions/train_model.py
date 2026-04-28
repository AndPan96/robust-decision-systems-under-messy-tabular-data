import torch
from src.training.dro import dro_train_epoch
from src.training.dro_folds import generate_inner_fold_loaders
from src.training.cross_validation import generate_outer_folds
from src.config.registry import MODEL_REGISTRY
from src.models.prior_model import PriorModel
from typing import cast
import numpy as np
from src.config.data_config import PLOT_STEPS, DRO_GROUPS
from src.actions.preprocess import preprocess_fit
import pandas as pd

class IdentityPreprocessor:
    def transform(self, X):
        return X
    
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

    model_class_arch = MODEL_REGISTRY[model_class.__name__]["arch_params"]  
    hyperparams = MODEL_REGISTRY[model_class.__name__]["train_params"]
    criterion = torch.nn.CrossEntropyLoss()
    metrics_store = {
        "loss": [],
        "accuracy": []
    }

    kf, imp_input_dim = generate_outer_folds(X, y, splits)
    training_curves = []
    validation_curves = []

    for X_train, y_train, X_val, y_val in kf:

        X_val = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val.values, dtype=torch.long).to(device)
        fold_train_steps = []
        fold_val_steps = []

        if model_class == PriorModel:
            model: torch.nn.Module = model_class(**model_class_arch).to(device)
            model.fit(torch.tensor(y_train.values, dtype=torch.long))
        else:
            model: torch.nn.Module = model_class(input_dim=imp_input_dim,**model_class_arch).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = hyperparams['lr'])
            fold_loaders = generate_inner_fold_loaders(X_train, y_train, DRO_GROUPS, batch_size = hyperparams['batch_size'])

            for curr_step in range(hyperparams['steps']):
                step_losses = dro_train_epoch(model, fold_loaders, optimizer, device)
                fold_train_steps.append(step_losses)

                if curr_step % PLOT_STEPS == 0:
                    model.eval()
                    with torch.no_grad():
                        logits = model(X_val)
                        loss = cast(torch.Tensor, criterion(logits, y_val)).item()
                        fold_val_steps.append((curr_step, loss))
                    model.train()

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            loss = cast(torch.Tensor, criterion(logits, y_val)).item()
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_val).float().mean().item()

        metrics_store["loss"].append(loss)
        metrics_store["accuracy"].append(accuracy)
        training_curves.append(fold_train_steps)
        validation_curves.append(fold_val_steps)


    if model_class == PriorModel:
        preprocessor = IdentityPreprocessor()
        model: torch.nn.Module = model_class(**model_class_arch).to(device)
        model.fit(torch.tensor(y.values, dtype=torch.long))
    else:
        model: torch.nn.Module = model_class(input_dim=imp_input_dim,**model_class_arch).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = hyperparams['lr'])
        X_imp, preprocessor = preprocess_fit(X)
        X_imp = cast(pd.DataFrame, X_imp)
        fold_loaders = generate_inner_fold_loaders(X_imp, y, DRO_GROUPS, batch_size = hyperparams['batch_size'])

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
        "preprocessor": preprocessor,
        "metrics": metrics,
        "training_curves": training_curves,
        "validation_curves": validation_curves,
        "input_dim": imp_input_dim,
        "hyperparams": hyperparams
    }