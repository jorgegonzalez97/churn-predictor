from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from great_expectations.dataset import PandasDataset
from pathlib import Path

DATA_DIR = Path("/opt/airflow/data")

def validate():
    df = pd.read_csv(DATA_DIR / "dataset.csv")
    gdf = PandasDataset(df)
    # simple assertions, customize later
    assert gdf.expect_column_to_exist("churn")["success"]
    assert gdf.expect_table_row_count_to_be_between(min_value=100, max_value=1000000)["success"]
    return True

with DAG(
    dag_id="churn_data_quality",
    start_date=datetime(2025,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["quality"]
) as dag:
    task = PythonOperator(task_id="validate_input", python_callable=validate)
