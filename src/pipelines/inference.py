from src.actions.predict_split import predict_split
from src.actions.predict import predict

def inference_pipeline():
    predict_split()
    predict()