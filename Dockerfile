FROM apache/airflow:2.9.2-python3.9
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY airflow/requirements.txt /requirements.txt

USER airflow
RUN pip install --no-cache-dir -r /requirements.txt \
    && pip install --no-cache-dir \
        apache-airflow-providers-amazon \
        apache-airflow-providers-postgres \
        apache-airflow-providers-http