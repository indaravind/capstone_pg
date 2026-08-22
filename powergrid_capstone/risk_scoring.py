import pandas as pd
from .config import TARGET_COL, IDENTIFIER_COLS


def compute_risk_scores(df: pd.DataFrame, best_pipe, output_dir: str) -> None:
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
    asset_path = f"{output_dir}/asset_risk_scores.csv"
    risk_df.to_csv(asset_path, index=False)
    agg = risk_df.groupby(["asset_type", "substation_region"]).agg(
        avg_failure_prob=("predicted_failure_prob", "mean"),
        total_expected_impact=("expected_impact", "sum"),
        avg_risk_score=("risk_score", "mean"),
    ).reset_index()
    agg_path = f"{output_dir}/risk_by_asset_type_region.csv"
    agg.to_csv(agg_path, index=False)
