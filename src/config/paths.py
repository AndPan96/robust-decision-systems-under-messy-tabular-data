from pathlib import Path

LOGS_FILE = Path("logs/app.log")

RAW_DIR = Path("data/raw")
RAW_APPLICATION_TRAIN = RAW_DIR / "application_train.csv"
RAW_BUREAU_BALANCE = RAW_DIR / "bureau_balance.csv"
RAW_BUREAU = RAW_DIR / "bureau.csv"
RAW_CREDIT_CARD_BALANCE = RAW_DIR / "credit_card_balance.csv"
RAW_INSTALLMENTS_PAYMENTS = RAW_DIR / "installments_payments.csv"
RAW_POS_CASH_BALANCE = RAW_DIR / "POS_CASH_balance.csv"
RAW_PREVIOUS_APPLICATION = RAW_DIR / "previous_application.csv"

DATASET_PATH = Path("data/raw/dataset.csv")
UPDATES_FILE = Path("data/incoming/updates.json")

TRAINING_DIR = Path("data/training")
TRAINING_X = TRAINING_DIR / "X_training.parquet"
TRAINING_Y = TRAINING_DIR / "y_training.parquet"

TEST_DIR = Path("data/test")
TEST_X = TEST_DIR / "X_test.parquet"
TEST_Y = TEST_DIR / "y_test.parquet"

MONITORING_DIR = Path("data/monitoring")
MONITORING_X = MONITORING_DIR / "X_mntr.parquet"
MONITORING_IDS = MONITORING_DIR / "ids_mntr.parquet"

PROCESSED_DIR = Path("data/processed")
PROCESSED_X = PROCESSED_DIR / "X_proc.parquet"
PROCESSED_IDS = PROCESSED_DIR / "ids_proc.parquet"
PROCESSED_PREDS = PROCESSED_DIR / "preds_proc.parquet"