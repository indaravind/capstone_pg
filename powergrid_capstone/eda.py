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
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def generate_eda_report(df: pd.DataFrame, output_path: str) -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating EDA report started at {start_time}")
    rows, cols = df.shape
    target_counts = df[TARGET_COL].value_counts(dropna=False)
    target_pct = df[TARGET_COL].value_counts(normalize=True, dropna=False)
    missing = df.isna().sum()
    lines = []
    lines.append("# PowerGrid EDA Report\n\n")
    lines.append("## Dataset Overview\n")
    lines.append(f"- Rows: {rows}\n")
    lines.append(f"- Columns: {cols}\n\n")
    lines.append("## Target Distribution (grid_failure_flag)\n")
    for k, v in target_counts.items():
        pct = target_pct[k]
        lines.append(f"- {k}: {v} ({pct:.3f})\n")
    lines.append("\n## Missing Values by Column\n")
    for col, mc in missing.items():
        if mc > 0:
            lines.append(f"- {col}: {mc} missing\n")
    with open(output_path, "w") as f:
        f.write("".join(lines))
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating EDA report ended at {end_time}")


def generate_eda_plots(df: pd.DataFrame, output_dir: str) -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating EDA plots started at {start_time}")

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    target_counts = df[TARGET_COL].value_counts()
    bars = ax.bar(["Normal Operating (0)", "Grid Failure (1)"], target_counts.values, color=["#2b5c8f", "#d9534f"], width=0.5)
    ax.set_title("Grid Asset Target Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Assets", fontsize=11)
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_target_distribution.png"), dpi=300)
    plt.close()

    # 2. Failure Rate by Asset Type
    if "asset_type" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        asset_stats = df.groupby("asset_type")[TARGET_COL].agg(["count", "mean"]).reset_index()
        asset_stats = asset_stats.sort_values(by="mean", ascending=False)
        bars = ax.barh(asset_stats["asset_type"], asset_stats["mean"] * 100, color="#337ab7", height=0.5)
        ax.set_title("Grid Failure Rate by Asset Type (%)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Failure Rate (%)", fontsize=11)
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.1f}%",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=10, fontweight='bold')
        ax.set_xlim(0, max(asset_stats["mean"] * 100) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_failure_rate_by_asset_type.png"), dpi=300)
        plt.close()

    # 3. Failure Rate by Region
    if "substation_region" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        region_stats = df.groupby("substation_region")[TARGET_COL].agg(["count", "mean"]).reset_index()
        region_stats = region_stats.sort_values(by="mean", ascending=False)
        bars = ax.bar(region_stats["substation_region"], region_stats["mean"] * 100, color="#5cb85c", width=0.5)
        ax.set_title("Grid Failure Rate by Substation Region (%)", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Failure Rate (%)", fontsize=11)
        ax.set_xlabel("Substation Region", fontsize=11)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(region_stats["mean"] * 100) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_failure_rate_by_region.png"), dpi=300)
        plt.close()

    # 4. Numeric Feature Correlation Heatmap (Top Correlations with Target)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in IDENTIFIER_COLS]
    if TARGET_COL in num_cols:
        corrs = df[num_cols].corr()[TARGET_COL].abs().sort_values(ascending=False)
        top_corr_cols = corrs.head(12).index.tolist()
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        corr_matrix = df[top_corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True, square=True,
                    annot_kws={"size": 9})
        ax.set_title("Correlation Heatmap of Key Features with Failure Flag", fontsize=13, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=300)
        plt.close()

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating EDA plots ended at {end_time}")

