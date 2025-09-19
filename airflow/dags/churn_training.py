from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import os, json, mlflow
from pathlib import Path

from include.scripts.prepare_data import prepare_dataset
from include.scripts.train import train as train_model
from include.scripts.register_model import promote_to_stage

DATA_DIR = Path("/opt/airflow/data")
TRAIN_OUT = DATA_DIR / "intermediate" / "train.csv"
TEST_OUT = DATA_DIR / "intermediate" / "test.csv"

default_args = {
    "owner": "mlops",
    "retries": 0
}

with DAG(
    dag_id="churn_training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["mlops","train"],
) as dag:

    def _prepare():
        input_csv = str(DATA_DIR / "dataset.csv")
        prepare_dataset(input_csv, str(TRAIN_OUT), str(TEST_OUT))

    def _train(**ctx):
        m = train_model(str(TRAIN_OUT), str(TEST_OUT))
        # record run id from last active run
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[client.get_experiment_by_name("churn_baseline").experiment_id],
                                  order_by=["attributes.start_time DESC"], max_results=1)
        run_id = runs[0].info.run_id if runs else ""
        ctx["ti"].xcom_push(key="run_id", value=run_id)
        return json.dumps(m)

    def _register(**ctx):
        promote_to_stage("telecom_churn_lr", "Staging")

    prepare = PythonOperator(task_id="prepare_data", python_callable=_prepare)
    train = PythonOperator(task_id="train_model", python_callable=_train)
    register = PythonOperator(task_id="register_model", python_callable=_register)

    prepare >> train >> register
