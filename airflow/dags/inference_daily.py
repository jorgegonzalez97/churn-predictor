from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd, mlflow, os
from pathlib import Path

DATA_DIR = Path("/opt/airflow/data")

def daily_inference():
    # Example inference reading same test set for demo
    df = pd.read_csv(DATA_DIR / "intermediate" / "test.csv")
    model_name = "telecom_churn_lr"
    stage = "Staging"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    preds = model.predict(df.drop(columns=["churn"]))
    out = df.copy()
    out["prediction"] = preds if preds.ndim == 1 else preds[:,1]
    out.to_csv(DATA_DIR / "predictions.csv", index=False)
    return str(DATA_DIR / "predictions.csv")

with DAG(
    dag_id="daily_inference",
    start_date=datetime(2025,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["inference"]
) as dag:
    task = PythonOperator(task_id="predict", python_callable=daily_inference)
