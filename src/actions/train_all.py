import torch
from typing import List, Dict, Any
from src.actions.train_model import train_model
from src.config.registry import MODEL_REGISTRY

def train_all(X, y, data_version, splits, device):

    results: List[Dict[str, Any]] = []

    for model_class in MODEL_REGISTRY.keys():

        result = train_model(MODEL_REGISTRY[model_class]["class"], X, y, data_version, 
                             splits, device)
        
        results.append(result)

    return results