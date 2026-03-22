from src.actions.monitor_model import monitor_model
from src.config.state import load_state, save_state
import logging

logger = logging.getLogger(__name__)

def monitoring_pipeline():
    result = monitor_model()
    
    if result is None:
        logger.info("Monitoring skipped: not enough labeled data yet")
        return
    
    logger.info(f"Monitoring accuracy: {result['accuracy']:.4f}")
    
    if result["retrain"]:
        state = load_state()
        state["retrain_required"] = True
        save_state(state)
        logger.info("Accuracy below threshold: retraining")