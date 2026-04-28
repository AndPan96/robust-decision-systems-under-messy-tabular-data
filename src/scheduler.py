from apscheduler.schedulers.blocking import BlockingScheduler
import logging
from logging.handlers import RotatingFileHandler
from src.config.paths import LOGS_FILE
from src.pipelines.inference import inference_pipeline
from src.pipelines.monitoring import monitoring_pipeline
from src.pipelines.retrain_all import retrain_all_pipeline
from src.pipelines.retrain_single import retrain_single_pipeline
from pipelines.up_rows import update_rows_pipeline

from src.config.state import load_state

handler = RotatingFileHandler(LOGS_FILE,maxBytes=5_000_000,backupCount=3)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[handler])

logger = logging.getLogger(__name__)
scheduler = BlockingScheduler()

def operating():
    state = load_state()

    if not state["retrain_required"]:
        
        update_rows_pipeline()
        inference_pipeline()
        monitoring_pipeline()

    else:

        if not state["retrain_single"]:
            retrain_single_pipeline()

        if not state["retrain_all"]:
            retrain_all_pipeline()


scheduler.add_job(operating, "interval", seconds=30)

def start_scheduler():
    logger.info("Starting Scheduler")
    scheduler.start()

if __name__ == "__main__":
    start_scheduler()
