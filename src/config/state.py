import json
from pathlib import Path

STATE_FILE = Path("data/state/state.json")

DEFAULT_STATE = {
    "retrain_required": False,
    "retrain_single": False,
    "retrain_all": False
}

def load_state():

    if not STATE_FILE.exists():
        save_state(DEFAULT_STATE)
        return DEFAULT_STATE.copy()
    
    with STATE_FILE.open("r") as f:
        state = json.load(f)

    return state

def save_state(state: dict):

    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent = 2)