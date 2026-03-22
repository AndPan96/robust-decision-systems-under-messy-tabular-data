import pytest
from src.config.state import load_state, save_state

def test_state():

    state = load_state()

    assert state["retrain_required"] == False
    assert state["retrain_single"] == False
    assert state["retrain_all"] == False

    state["retrain_required"] = True
    save_state(state)
    state = load_state()

    assert state["retrain_required"] == True