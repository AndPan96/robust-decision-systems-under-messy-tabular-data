from src.actions.monitor_model import monitor_model
from src.actions.transform_data import transform_data
from src.config.state import load_state, save_state
import logging
from src.config.registry import load_current_model
import pandas as pd
from typing import cast
from pathlib import Path
from src.utils.plots import generate_monitoring_plot
from src.utils.reports import generate_monitoring_report

logger = logging.getLogger(__name__)

def monitoring_pipeline():
    needed = transform_data()
    if needed:
        logger.info("ELT operations performed from data/raw tables to data/raw/dataset.csv.")
    else:
        logger.info("No transformation needed: ELT operations skipped.")

    result = monitor_model()
    
    if result is None:
        logger.info("Monitoring skipped: not enough labeled data yet")
        return
    
    logger.info(f"Monitoring accuracy: {result['accuracy']:.4f}")

    _, _, meta = load_current_model()
    report_path = cast(Path, meta["report_path"])
    plots_path = cast(Path, meta["plots_path"])
    metrics_path = report_path / "metrics.parquet"

    if metrics_path.exists():
        df = pd.read_parquet(metrics_path)
    else:
        df = pd.DataFrame(columns=["timestamp", "accuracy", "n_samples"])
    new_row = {
        "timestamp": pd.Timestamp.now(),
        "accuracy": result["accuracy"],
        "n_samples": result["n_samples"]
    }
    df.loc[len(df)] = new_row
    df.to_parquet(metrics_path)

    generate_monitoring_plot(metrics_path, plots_path / "monitoring.png")
    logger.info(f"Monitoring plots saved in {plots_path}.")

    generate_monitoring_report(report_path)
    logger.info(f"Report of {report_path} monitoring updated.")
    
    if result["retrain"]:
        state = load_state()
        state["retrain_required"] = True
        save_state(state)
        logger.info("Accuracy below threshold: retraining")