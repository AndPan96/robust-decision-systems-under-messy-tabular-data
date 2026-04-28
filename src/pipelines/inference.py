from src.actions.transform_data import transform_data
from src.actions.predict_split import predict_split
from src.actions.predict import predict
import logging

logger = logging.getLogger(__name__)

def inference_pipeline():
    needed = transform_data()
    if needed:
        logger.info("ELT operations performed from data/raw tables to data/raw/dataset.csv.")
    else:
        logger.info("No transformation needed: ELT operations skipped.")

    predict_split()
    logger.info("Unlabeled and unprocessed data loaded to data/monitoring.")
    
    predict()
    logger.info("Data in data/monitoring processed and loaded to data/processed.")