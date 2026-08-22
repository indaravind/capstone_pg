import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .config import TARGET_COL, IDENTIFIER_COLS


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


def train_all_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, Pipeline]:
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
    return metrics_df, best_model_name, best_pipe


def save_model_metrics(metrics_df: pd.DataFrame, path: str) -> None:
    metrics_df.to_csv(path, index=False)


def generate_model_evaluation_report(metrics_df: pd.DataFrame, best_model_name: str, path: str) -> None:
    lines = []
    lines.append("# Model Evaluation Report")
    lines.append("## Summary Metrics")
    for _, row in metrics_df.iterrows():
        lines.append(f"### {row['model']}")
        lines.append(f"- Accuracy: {row['accuracy']:.4f}\n")
        lines.append(f"- Precision: {row['precision']:.4f}\n")
        lines.append(f"- Recall: {row['recall']:.4f}\n")
        lines.append(f"- F1-score: {row['f1']:.4f}\n")
        lines.append(f"- ROC-AUC: {row['roc_auc']:.4f}\n")
    lines.append("## Best Model")
    lines.append(f"- Selected model: {best_model_name} based on highest ROC-AUC\n")
    with open(path, "w") as f:
        f.write("".join(lines))


def compute_feature_importance(best_pipe: Pipeline, path: str) -> None:
    clf = best_pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    try:
        feature_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]
    fi_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    fi_df.to_csv(path, index=False)
