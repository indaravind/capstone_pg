import logging
from datetime import datetime
import pandas as pd
from sklearn.impute import SimpleImputer
from .config import TARGET_COL, IDENTIFIER_COLS

logger = logging.getLogger(__name__)


def load_raw_dataset(path: str) -> pd.DataFrame:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Loading raw dataset started at {start_time}")
    df = pd.read_csv(path, low_memory=False)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Loading raw dataset ended at {end_time}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Cleaning dataset started at {start_time}")
    df = df.dropna(subset=[TARGET_COL])
    clean_df = df.copy()
    
    # Standardize string formatting and case variations (e.g., converting 'TRANSFORMER' and 'transformer' to 'Transformer')
    if "asset_type" in clean_df.columns:
        clean_df["asset_type"] = clean_df["asset_type"].apply(
            lambda x: x.strip().title() if isinstance(x, str) else x
        )
    for col in clean_df.select_dtypes(include=["object"]).columns:
        clean_df[col] = clean_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    feature_cols = [c for c in clean_df.columns if c not in IDENTIFIER_COLS + [TARGET_COL]]
    numeric_features = clean_df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    
    if numeric_features:
        num_imp = SimpleImputer(strategy="median")
        clean_df[numeric_features] = num_imp.fit_transform(clean_df[numeric_features])
    if categorical_features:
        cat_imp = SimpleImputer(strategy="most_frequent")
        clean_df[categorical_features] = cat_imp.fit_transform(clean_df[categorical_features])
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Cleaning dataset ended at {end_time}")
    return clean_df


def save_clean_dataset(df: pd.DataFrame, path: str) -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Saving clean dataset started at {start_time}")
    df.to_csv(path, index=False)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Saving clean dataset ended at {end_time}")

