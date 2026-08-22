# PowerGrid Capstone Project

## Structure

- powergrid_capstone/
  - __init__.py
  - config.py
  - data_preprocessing.py
  - eda.py
  - modeling.py
  - risk_scoring.py
  - main.py
- data/
  - PowerGrid_Utility_Intelligence_Dataset_10k.csv
- notebooks/
  - PowerGrid_capstone_notebook.ipynb
- artifacts/
  - (generated outputs: cleaned dataset, reports, metrics, risk scores)

## How to run

1. Create a Python environment with the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. From the project root (this folder), run the main script:

   ```bash
   python -m powergrid_capstone.main
   ```

3. After it finishes, check the `artifacts/` directory for:

   - PowerGrid_cleaned_dataset.csv
   - EDA_report.md
   - model_metrics.csv
   - model_evaluation_report.md
   - random_forest_feature_importance.csv
   - asset_risk_scores.csv
   - risk_by_asset_type_region.csv

4. Alternatively, open the notebook from `notebooks/PowerGrid_capstone_notebook.ipynb` and run the cells.

The package uses `config.py` to locate the dataset in `data/` and writes all outputs step by step into `artifacts/`.
