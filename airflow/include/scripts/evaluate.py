import json, pandas as pd, numpy as np, mlflow, os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate(run_id: str, test_csv: str):
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.read_csv(test_csv)
    y = df.iloc[:, -1].astype(int) if df.dtypes[-1] != 'O' else df.iloc[:, -1].astype(int)
    X = df.drop(columns=[df.columns[-1]])
    proba = model.predict(X)
    if isinstance(proba, np.ndarray) and proba.ndim > 1 and proba.shape[1] == 2:
        proba = proba[:, 1]
    preds = (proba >= 0.5).astype(int) if isinstance(proba, np.ndarray) else model.predict(X)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds).tolist()
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)
    return {"classification_report": report, "confusion_matrix": cm, "roc_auc": float(roc_auc)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--test_csv", required=True)
    args = ap.parse_args()
    out = evaluate(args.run_id, args.test_csv)
    print(json.dumps(out))
