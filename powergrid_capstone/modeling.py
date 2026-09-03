import logging
import os
from datetime import datetime
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from .config import TARGET_COL, IDENTIFIER_COLS

logger = logging.getLogger(__name__)

# Set global visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'


def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list, list]:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocess, numeric_features, categorical_features, numeric_features + categorical_features


def train_all_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, Pipeline, Dict[str, Pipeline], pd.DataFrame, pd.Series]:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Training all models started at {start_time}")
    feature_cols = [c for c in df.columns if c not in IDENTIFIER_COLS + [TARGET_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL]
    preprocess, _, _, _ = _build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    }
    metrics_rows = []
    trained_pipelines: Dict[str, Pipeline] = {}
    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        metrics_rows.append(row)
        trained_pipelines[name] = pipe
    metrics_df = pd.DataFrame(metrics_rows)
    best_row = metrics_df.sort_values("roc_auc", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_pipe = trained_pipelines[best_model_name]
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Training all models ended at {end_time}")
    return metrics_df, best_model_name, best_pipe, trained_pipelines, X_test, y_test


def save_model_metrics(metrics_df: pd.DataFrame, path: str) -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Saving model metrics started at {start_time}")
    metrics_df.to_csv(path, index=False)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Saving model metrics ended at {end_time}")


def generate_model_evaluation_report(metrics_df: pd.DataFrame, best_model_name: str, path: str) -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating model evaluation report started at {start_time}")
    lines = []
    lines.append("# Model Evaluation Report\n\n")
    lines.append("## Summary Metrics\n")
    for _, row in metrics_df.iterrows():
        lines.append(f"### {row['model']}\n")
        lines.append(f"- Accuracy: {row['accuracy']:.4f}\n")
        lines.append(f"- Precision: {row['precision']:.4f}\n")
        lines.append(f"- Recall: {row['recall']:.4f}\n")
        lines.append(f"- F1-score: {row['f1']:.4f}\n")
        lines.append(f"- ROC-AUC: {row['roc_auc']:.4f}\n\n")
    lines.append("## Best Model\n")
    lines.append(f"- Selected model: {best_model_name} based on highest ROC-AUC\n")
    with open(path, "w") as f:
        f.write("".join(lines))
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating model evaluation report ended at {end_time}")


def compute_feature_importance(best_pipe: Pipeline, path: str) -> pd.DataFrame:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Computing feature importance started at {start_time}")
    clf = best_pipe.named_steps["clf"]
    fi_df = pd.DataFrame()
    if hasattr(clf, "feature_importances_"):
        try:
            feature_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
            # Clean feature names (remove cat__ num__ prefixes for better readability)
            clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
        except Exception:
            clean_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]
        fi_df = pd.DataFrame({"feature": clean_names, "importance": clf.feature_importances_}).sort_values(
            "importance", ascending=False
        )
        fi_df.to_csv(path, index=False)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Computing feature importance ended at {end_time}")
    return fi_df


def generate_model_plots(trained_pipelines: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series, best_model_name: str, output_dir: str) -> None:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating model plots started at {start_time}")

    # 1. Comparative ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = {
        "RandomForest": "#2b5c8f",
        "SVM": "#5cb85c",
        "LogisticRegression": "#f0ad4e",
        "DecisionTree": "#d9534f"
    }
    for name, pipe in trained_pipelines.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        color = colors.get(name, "#333333")
        lw = 2.5 if name == best_model_name else 1.5
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", color=color, linewidth=lw)

    ax.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5000)", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title("ROC Curves Comparison Across Machine Learning Models", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_roc_curves.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrix for Best Model
    best_pipe = trained_pipelines[best_model_name]
    y_pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Normal (0)", "Failure (1)"],
                yticklabels=["Normal (0)", "Failure (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix: {best_model_name} (Best Model)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_confusion_matrix.png"), dpi=300)
    plt.close()

    # 3. Top Feature Importance Chart
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        try:
            feature_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
            clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
        except Exception:
            clean_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]
        fi_df = pd.DataFrame({"feature": clean_names, "importance": clf.feature_importances_}).sort_values(
            "importance", ascending=False
        ).head(15)

        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        bars = ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#2b5c8f", height=0.6)
        ax.set_title(f"Top 15 Predictive Features ({best_model_name})", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Feature Importance Score", fontsize=11)
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.4f}",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9)
        ax.set_xlim(0, max(fi_df["importance"]) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_feature_importance.png"), dpi=300)
        plt.close()

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Generating model plots ended at {end_time}")

