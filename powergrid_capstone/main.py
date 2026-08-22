import os
from .config import RAW_DATA_PATH, OUTPUT_DIR
from .data_preprocessing import load_raw_dataset, clean_dataset, save_clean_dataset
from .eda import generate_eda_report
from .modeling import train_all_models, save_model_metrics, generate_model_evaluation_report, compute_feature_importance
from .risk_scoring import compute_risk_scores


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_df = load_raw_dataset(RAW_DATA_PATH)
    clean_df = clean_dataset(raw_df)
    clean_path = os.path.join(OUTPUT_DIR, "PowerGrid_cleaned_dataset.csv")
    save_clean_dataset(clean_df, clean_path)
    eda_path = os.path.join(OUTPUT_DIR, "EDA_report.md")
    generate_eda_report(clean_df, eda_path)
    metrics_df, best_model_name, best_pipe = train_all_models(clean_df)
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    save_model_metrics(metrics_df, metrics_path)
    eval_path = os.path.join(OUTPUT_DIR, "model_evaluation_report.md")
    generate_model_evaluation_report(metrics_df, best_model_name, eval_path)
    fi_path = os.path.join(OUTPUT_DIR, "random_forest_feature_importance.csv")
    compute_feature_importance(best_pipe, fi_path)
    compute_risk_scores(clean_df, best_pipe, OUTPUT_DIR)


if __name__ == "__main__":
    main()
