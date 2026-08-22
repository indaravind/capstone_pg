import pandas as pd
from .config import TARGET_COL


def generate_eda_report(df: pd.DataFrame, output_path: str) -> None:
    rows, cols = df.shape
    target_counts = df[TARGET_COL].value_counts(dropna=False)
    target_pct = df[TARGET_COL].value_counts(normalize=True, dropna=False)
    missing = df.isna().sum()
    lines = []
    lines.append("# PowerGrid EDA Report")
    lines.append("## Dataset Overview")
    lines.append(f"- Rows: {rows}\n")
    lines.append(f"- Columns: {cols}\n")
    lines.append("## Target Distribution (grid_failure_flag)")
    for k, v in target_counts.items():
        pct = target_pct[k]
        lines.append(f"- {k}: {v} ({pct:.3f})\n")
    lines.append("## Missing Values by Column")
    for col, mc in missing.items():
        if mc > 0:
            lines.append(f"- {col}: {mc} missing\n")
    with open(output_path, "w") as f:
        f.write("".join(lines))
