from src.actions.transform_data import transform_data
from src.actions.train_split import train_split
from src.actions.test_deploy_model import test_deploy_model

def retrain_single_pipeline():
    transform_data()
    train_split()
    test_deploy_model()