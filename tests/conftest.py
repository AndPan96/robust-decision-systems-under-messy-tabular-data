from pathlib import Path
import pytest
import pandas as pd
import shutil
from src.config.paths import TRAINING_DIR, TEST_DIR, MONITORING_DIR, PROCESSED_DIR, \
        DATASET_PATH, RAW_APPLICATION_TRAIN, RAW_BUREAU, RAW_BUREAU_BALANCE, \
        RAW_CREDIT_CARD_BALANCE, RAW_INSTALLMENTS_PAYMENTS, RAW_POS_CASH_BALANCE, RAW_PREVIOUS_APPLICATION
from src.config.state import STATE_FILE

@pytest.fixture(autouse=True)
def clean_state():
    STATE_FILE.unlink(missing_ok=True)
    yield
    STATE_FILE.unlink(missing_ok=True)

def clean_folder(folder: Path):
    if folder.exists():
        for f in folder.iterdir():
            if f.is_file():
                f.unlink()


@pytest.fixture
def set_environment():
    
    def _set_environment():
        test_dataset_path = Path("data/application_train.csv")
        shutil.copy(test_dataset_path,DATASET_PATH)
        test_app_train_path = Path("data/application_train.csv")
        shutil.copy(test_app_train_path,RAW_APPLICATION_TRAIN)
        test_bb_path = Path("data/bureau_balance.csv")
        shutil.copy(test_bb_path,RAW_BUREAU_BALANCE)
        test_bureau_path = Path("data/bureau.csv")
        shutil.copy(test_bureau_path,RAW_BUREAU)
        test_ccb_path = Path("data/credit_card_balance.csv")
        shutil.copy(test_ccb_path,RAW_CREDIT_CARD_BALANCE)
        test_ip_path = Path("data/installments_payments.csv")
        shutil.copy(test_ip_path,RAW_INSTALLMENTS_PAYMENTS)
        test_pcb_path = Path("data/POS_CASH_balance.csv")
        shutil.copy(test_pcb_path,RAW_POS_CASH_BALANCE)
        test_papp_path = Path("data/previous_application.csv")
        shutil.copy(test_papp_path,RAW_PREVIOUS_APPLICATION)

        clean_folder(TRAINING_DIR)
        clean_folder(TEST_DIR)
        clean_folder(MONITORING_DIR)
        clean_folder(PROCESSED_DIR)
        clean_folder(Path("saved_models"))

    return _set_environment

@pytest.fixture
def reset_environment():

    def _reset_environment():
        DATASET_PATH.unlink()
        clean_folder(DATASET_PATH)
        RAW_APPLICATION_TRAIN.unlink()
        clean_folder(RAW_APPLICATION_TRAIN)
        RAW_BUREAU_BALANCE.unlink()
        clean_folder(RAW_BUREAU_BALANCE)
        RAW_BUREAU.unlink()
        clean_folder(RAW_BUREAU)
        RAW_CREDIT_CARD_BALANCE.unlink()
        clean_folder(RAW_CREDIT_CARD_BALANCE)
        RAW_INSTALLMENTS_PAYMENTS.unlink()
        clean_folder(RAW_INSTALLMENTS_PAYMENTS)
        RAW_POS_CASH_BALANCE.unlink()
        clean_folder(RAW_POS_CASH_BALANCE)
        RAW_PREVIOUS_APPLICATION.unlink()
        clean_folder(RAW_PREVIOUS_APPLICATION)
        
        clean_folder(TRAINING_DIR)
        clean_folder(TEST_DIR)
        clean_folder(MONITORING_DIR)
        clean_folder(PROCESSED_DIR)
        clean_folder(Path("saved_models"))

    return _reset_environment

@pytest.fixture
def fake_dataset(tmp_path):

    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4],
        "F1": [.1, .2, .3, .4],
        "F2": [1, 2, 3, 4],
        "TARGET": [1, 0, None, None]
    })

    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)

    return path