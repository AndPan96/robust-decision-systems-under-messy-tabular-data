from src.actions.transform_data import transform_data
from src.actions.train_split import train_split
from src.actions.test_deploy_all import test_deploy_all
from src.utils.plots import generate_train_plot, plot_model_comparison
from src.utils.reports import create_new_report_folder, generate_train_report, REPORT_DIR
import logging

logger = logging.getLogger(__name__)

def retrain_all_pipeline():
    needed = transform_data()
    if needed:
        logger.info("ELT operations performed from data/raw tables to data/raw/dataset.csv.")
    else:
        logger.info("No transformation needed: ELT operations skipped.")

    train_split()
    logger.info("Labeled data loaded to data/train and data/test.")

    results = test_deploy_all()
    logger.info("Training of all models completed.")

    report_path, plots_path = create_new_report_folder(REPORT_DIR)
    logger.info(f"Training report folder {report_path} created.")

    plot_model_comparison(results, plots_path / "comparison.png")
    for result in results:
        generate_train_plot(result, plots_path / f"{result["name"]}.png")
    logger.info(f"Training plots saved in {plots_path}.")
    
    generate_train_report(results, report_path)
    logger.info(f"Report of {report_path} training generated.")
    