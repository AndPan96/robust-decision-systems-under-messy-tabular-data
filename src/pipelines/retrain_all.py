from src.actions.train_split import train_split
from src.actions.test_deploy_all import test_deploy_all

def retrain_all_pipeline():
    train_split()
    test_deploy_all()