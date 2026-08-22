import pandas as pd
from sklearn.impute import SimpleImputer
from .config import TARGET_COL, IDENTIFIER_COLS


def load_raw_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=[TARGET_COL])
    feature_cols = [c for c in df.columns if c not in IDENTIFIER_COLS + [TARGET_COL]]
    numeric_features = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    clean_df = df.copy()
    if numeric_features:
        num_imp = SimpleImputer(strategy="median")
        clean_df[numeric_features] = num_imp.fit_transform(clean_df[numeric_features])
    if categorical_features:
        cat_imp = SimpleImputer(strategy="most_frequent")
        clean_df[categorical_features] = cat_imp.fit_transform(clean_df[categorical_features])
    return clean_df


def save_clean_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
