import logging
from datetime import datetime
import os
from .config import RAW_DATA_PATH, OUTPUT_DIR
from .data_preprocessing import load_raw_dataset, clean_dataset, save_clean_dataset
from .eda import generate_eda_report, generate_eda_plots
from .modeling import train_all_models, save_model_metrics, generate_model_evaluation_report, compute_feature_importance, generate_model_plots
from .risk_scoring import compute_risk_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main() -> None:
    main_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Main execution started at {main_start}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_df = load_raw_dataset(RAW_DATA_PATH)
    clean_df = clean_dataset(raw_df)
    clean_path = os.path.join(OUTPUT_DIR, "PowerGrid_cleaned_dataset.csv")
    save_clean_dataset(clean_df, clean_path)
    
    # Generate EDA text report & visual plots
    eda_path = os.path.join(OUTPUT_DIR, "EDA_report.md")
    generate_eda_report(clean_df, eda_path)
    generate_eda_plots(clean_df, OUTPUT_DIR)

    # Train models & evaluate
    metrics_df, best_model_name, best_pipe, trained_pipelines, X_test, y_test = train_all_models(clean_df)
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    save_model_metrics(metrics_df, metrics_path)
    eval_path = os.path.join(OUTPUT_DIR, "model_evaluation_report.md")
    generate_model_evaluation_report(metrics_df, best_model_name, eval_path)
    
    # Feature importance & diagnostic plots
    fi_path = os.path.join(OUTPUT_DIR, "random_forest_feature_importance.csv")
    compute_feature_importance(best_pipe, fi_path)
    generate_model_plots(trained_pipelines, X_test, y_test, best_model_name, OUTPUT_DIR)

    # Risk scoring & financial risk plots
    compute_risk_scores(clean_df, best_pipe, OUTPUT_DIR)

    main_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Main execution ended at {main_end}")


if __name__ == "__main__":
    main()

