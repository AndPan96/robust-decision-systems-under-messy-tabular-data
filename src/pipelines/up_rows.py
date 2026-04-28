from src.actions.update_rows import update_rows
from src.config.paths import UPDATES_FILE
import json
import logging

logger = logging.getLogger(__name__)

def update_rows_pipeline():
    if not UPDATES_FILE.exists():
        logger.info(f"No update file to process.")
        return
    
    with open(UPDATES_FILE) as f:
        updates = json.load(f)
    update_rows(updates)
    logger.info(f"Tables in data/raw updated.")

    UPDATES_FILE.unlink()
    logger.info(f"Processed update file deleted.")


