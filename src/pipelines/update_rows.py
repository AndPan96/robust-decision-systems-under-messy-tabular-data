from src.actions.update_rows import update_rows
from src.config.paths import UPDATES_FILE
import json

def update_rows_pipeline():
    if not UPDATES_FILE.exists():
        return
    with open(UPDATES_FILE) as f:
        updates = json.load(f)
    update_rows(updates)
    UPDATES_FILE.unlink()


