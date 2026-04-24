from src.actions.transform_data import transform_data
from src.actions.predict_split import predict_split
from src.actions.predict import predict

def inference_pipeline():
    transform_data()
    predict_split()
    predict()