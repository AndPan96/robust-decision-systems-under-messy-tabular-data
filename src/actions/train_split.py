from src.data.load_data import load_dataset
from src.data.split import initial_split
import pandas as pd
from src.config.data_config import TRAIN_WINDOW, TEST_WINDOW
from src.config.paths import DATASET_PATH, TRAINING_X, TRAINING_Y, TEST_X, TEST_Y

def train_split():

    _, X, y = load_dataset(DATASET_PATH)

    print("Dataset loaded")
    print("Samples:", len(X))
    print("Features:", X.shape[1],"\n")

    X_train, y_train, X_test, y_test = initial_split(X, y, TRAIN_WINDOW, TEST_WINDOW)

    X_train.to_parquet(TRAINING_X)
    y_train.to_frame(name="TARGET").to_parquet(TRAINING_Y)

    X_test.to_parquet(TEST_X)
    y_test.to_frame(name="TARGET").to_parquet(TEST_Y)

    print("Split completed")
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))