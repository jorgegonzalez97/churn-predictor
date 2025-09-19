import os, yaml, mlflow, pandas as pd, numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train(train_csv: str, test_csv: str):
    cfg = load_config()
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    target = cfg.get("target_column", "churn")
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target].astype(int if df_train[target].dtype != 'O' else 'int')
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target].astype(int if df_test[target].dtype != 'O' else 'int')

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(transformers=[
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ])

    model = LogisticRegression(max_iter=1000, n_jobs=None)

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg.get("experiment_name", "churn_baseline"))
    with mlflow.start_run():
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else preds

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, proba)) if hasattr(pipe, "predict_proba") else float('nan'),
            "f1": float(f1_score(y_test, preds)),
            "accuracy": float(accuracy_score(y_test, preds))
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(pipe, "model", registered_model_name="telecom_churn_lr")

        # Tag run
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("pipeline", "baseline")
        return metrics

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    args = ap.parse_args()
    m = train(args.train_csv, args.test_csv)
    print(json.dumps(m))
