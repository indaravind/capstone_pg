import logging
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import TARGET_COL, IDENTIFIER_COLS

logger = logging.getLogger(__name__)

# Set global style for visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'


def compute_risk_scores(df: pd.DataFrame, best_pipe, output_dir: str) -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Computing risk scores started at {start_time}")
    feature_cols = [c for c in df.columns if c not in IDENTIFIER_COLS + [TARGET_COL]]
    X = df[feature_cols]
    proba = best_pipe.predict_proba(X)[:, 1]
    impact = df["estimated_revenue_loss"] + df["regulatory_penalty_cost"]
    risk_score = proba * impact
    risk_df = pd.DataFrame({
        "asset_id": df["asset_id"],
        "asset_type": df["asset_type"],
        "substation_region": df["substation_region"],
        TARGET_COL: df[TARGET_COL],
        "predicted_failure_prob": proba,
        "expected_impact": impact,
        "risk_score": risk_score,
    })
    asset_path = os.path.join(output_dir, "asset_risk_scores.csv")
    risk_df.to_csv(asset_path, index=False)
    agg = risk_df.groupby(["asset_type", "substation_region"]).agg(
        avg_failure_prob=("predicted_failure_prob", "mean"),
        total_expected_impact=("expected_impact", "sum"),
        avg_risk_score=("risk_score", "mean"),
    ).reset_index()
    agg_path = os.path.join(output_dir, "risk_by_asset_type_region.csv")
    agg.to_csv(agg_path, index=False)
    
    generate_risk_plots(agg, risk_df, output_dir)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Computing risk scores ended at {end_time}")


def generate_risk_plots(agg: pd.DataFrame, risk_df: pd.DataFrame, output_dir: str) -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating risk plots started at {start_time}")

    # Risk Score Exposure by Region and Asset Type
    pivot_risk = agg.pivot(index="substation_region", columns="asset_type", values="avg_risk_score")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    pivot_risk.plot(kind="bar", stacked=False, ax=ax, colormap="viridis", width=0.7)
    ax.set_title("Average Financial Risk Score ($) by Region and Asset Type", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Average Risk Exposure ($)", fontsize=11)
    ax.set_xlabel("Substation Region", fontsize=11)
    ax.legend(title="Asset Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_financial_impact_distribution.png"), dpi=300)
    plt.close()

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating risk plots ended at {end_time}")

