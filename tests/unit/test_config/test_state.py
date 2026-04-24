import pytest
from src.config.state import load_state, save_state
from src.config.state import STATE_FILE

def test_state():

    state = load_state()

    assert state["retrain_required"] == False
    assert state["retrain_single"] == False
    assert state["retrain_all"] == False
    assert state["transform_required"] == True

    state["retrain_required"] = True
    save_state(state)
    state = load_state()

    assert state["retrain_required"] == True

    STATE_FILE.unlink(missing_ok=True)