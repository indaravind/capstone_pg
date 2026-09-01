import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'artifacts')

TARGET_COL = "grid_failure_flag"
IDENTIFIER_COLS = [
    "asset_id",
    "legacy_asset_code",
    "monitoring_batch_id",
    "administrative_reference",
]
RAW_DATA_PATH = os.path.join(DATA_DIR, "PowerGrid_Utility_Intelligence.csv")
#RAW_DATA_PATH = os.path.join(DATA_DIR, "PowerGrid_Utility_Intelligence_Dataset_10k.csv")
OUTPUT_DIR = ARTIFACTS_DIR
