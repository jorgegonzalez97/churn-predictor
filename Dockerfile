FROM apache/airflow:2.9.2-python3.9
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    -c https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-3.9.txt

USER airflow

