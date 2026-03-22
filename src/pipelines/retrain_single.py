from src.actions.train_split import train_split
from src.actions.test_deploy_model import test_deploy_model

def retrain_single_pipeline():
    train_split()
    test_deploy_model()