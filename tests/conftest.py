from pathlib import Path
import pytest
import pandas as pd
import shutil
from src.config.paths import TRAINING_DIR, TEST_DIR, MONITORING_DIR, PROCESSED_DIR, STATE_PATH, DATASET_PATH

def clean_folder(folder: Path):
    if folder.exists():
        for f in folder.iterdir():
            if f.is_file():
                f.unlink()


@pytest.fixture(autouse=True)
def reset_environment():
    
    test_csv_path = Path("data/application_train.csv")
    shutil.copy(test_csv_path,DATASET_PATH)
    clean_folder(TRAINING_DIR)
    clean_folder(TEST_DIR)
    clean_folder(MONITORING_DIR)
    clean_folder(PROCESSED_DIR)
    clean_folder(STATE_PATH.parent)
    clean_folder(Path("saved_models"))

    yield

    DATASET_PATH.unlink()
    clean_folder(DATASET_PATH)
    clean_folder(TRAINING_DIR)
    clean_folder(TEST_DIR)
    clean_folder(MONITORING_DIR)
    clean_folder(PROCESSED_DIR)
    clean_folder(STATE_PATH.parent)
    clean_folder(Path("saved_models"))

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