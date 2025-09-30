# Stack MLOps de Churn con Airflow, MLflow y MinIO

Pipeline de churn listo para producción orquestado con Airflow y Docker Compose. La plataforma integra Airflow, Postgres, MLflow y MinIO y suministra flujos de referencia para calidad de datos, entrenamiento de modelos, scoring batch y monitoreo continuo.

## 1. Panorama del Stack

| Servicio | Propósito | Hostname | Puerto expuesto |
|----------|-----------|----------|-----------------|
| Airflow  | Orquestación (scheduler, webserver, LocalExecutor) | `airflow` | `8080` (mapeado a `127.0.0.1:18080`) |
| Postgres | Metadatos de Airflow, backend de MLflow, tablas de monitoreo | `postgres` | `5432` |
| MLflow   | Tracking de experimentos y Model Registry | `mlflow` | `5000` (mapeado a `127.0.0.1:5001`) |
| MinIO    | Almacenamiento de artefactos y modelos (compatible S3) | `minio` | `9000` + consola `9001` |

**Buckets** que crea `scripts/bootstrap_minio.sh`:

- `mlflow` (artefactos de MLflow)
- `models` (pipelines `.joblib` versionados)
- `reports` (PSI, monitoreo, reportes de drift en HTML/PDF)
- `predictions` (salidas de scoring batch)

## 2. Estructura del Proyecto

```
.
├─ airflow/
│  └─ dags/
│     ├─ dag_data_quality_psi.py
│     ├─ dag_train_register.py
│     ├─ dag_predict_new_data.py
│     └─ dag_monitoring.py
├─ src/
│  ├─ config.py
│  ├─ features/
│  ├─ models/
│  ├─ monitoring/
│  ├─ io/
│  └─ utils/
├─ scripts/bootstrap_minio.sh
├─ sql/create_monitoring_tables.sql
├─ requirements.txt
└─ data/
   ├─ dataset.csv
   └─ data_descriptions.csv
```

Módulos principales:

- `src/features`: inferencia de esquema, preprocesamiento, generación de PSI y reportes de calidad.
- `src/models`: tuning con Optuna, entrenamiento + registro en MLflow, inferencia batch.
- `src/monitoring`: reportes de drift, métricas post-despliegue, análisis por segmentos.
- `src/io`: wrappers ligeros para MinIO, MLflow y Postgres.

En `src/data` hay symlinks hacia los CSV canónicos ubicados en `data/`.

## 3. Prerrequisitos

1. Copia el ejemplo de variables de entorno y ajusta secretos:
   ```bash
   cp .env.example .env
   # actualiza AIRFLOW_UID, contraseñas, credenciales de MinIO, etc.
   ```
2. (Opcional) Crea un entorno virtual si ejecutarás scripts localmente:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 4. Levantar el Stack

```bash
docker compose up -d
```

Asistentes de primera ejecución (se automatizan al levantar `docker compose up`):

- `minio-setup` usa `mc` dentro de un contenedor efímero para crear los buckets `mlflow`, `models`, `reports` y `predictions`. Para relanzarlo manualmente:
  ```bash
  docker compose run --rm minio-setup
  ```
- `postgres-monitoring-init` aplica `sql/create_monitoring_tables.sql` contra la base `airflow`. Para reejecutarlo manualmente:
  ```bash
  docker compose run --rm postgres-monitoring-init
  ```

## 5. DAGs y Pipelines

| DAG | Programación | Propósito | Salidas principales |
|-----|--------------|-----------|---------------------|
| `dag_data_quality_psi` | `@daily` | Garantiza snapshot baseline, calcula PSI y calidad de datos, genera HTML/PDF | Reportes en MinIO `reports/`, run `data_quality_psi` en MLflow |
| `dag_train_register` | `@monthly` | Prepara datos, hace tuning con Optuna, entrena el mejor modelo y lo registra | Modelo en MLflow (Staging/Production), artefactos en MinIO `models/` |
| `dag_predict_new_data` | `@daily` | Scoring batch sobre `data/new_data.csv`, registra métricas en Postgres y dispara reentrenos | Predicciones en MinIO `predictions/`, registros en `monitoring.scoring_runs` |
| `dag_monitoring` | `@weekly` | Revisa drift con Evidently, monitorea performance y errores por segmento | Reportes en MinIO `reports/monitoring`, run `monitoring_metrics` en MLflow |

### Ejecuciones manuales (UI de Airflow)

1. Accede a http://localhost:18080 con las credenciales de `.env`.
2. Despausa el DAG deseado y lanza una ejecución manual.
3. Consulta artefactos y métricas en http://localhost:5001 (MLflow).
4. Consola de MinIO disponible en http://localhost:9001.

## 6. Datos y Baselines

- `src/features/psi.extract_baseline_if_missing` crea `data/baseline.parquet` la primera vez con el `BASELINE_SHARE` configurado (por defecto `0.2`).
- Los reportes de PSI se guardan en `reports/` local y en MinIO bajo `reports/<run_id>/`.
- Los pipelines de preprocesamiento se serializan como `.joblib` en `models/` y se replican en MinIO `models/pipelines/`.

## 7. Tablas de Monitoreo

Se crea el esquema ejecutando automáticamente `sql/create_monitoring_tables.sql`. Si necesitas revisarlo:

```sql
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE TABLE monitoring.scoring_runs (...);
CREATE TABLE monitoring.segment_metrics (...);
```

Airflow usa `src/io/postgres_logging` para registrar ejecuciones y métricas segmentadas.

## 8. Configuración

Variables relevantes (decláralas en `.env`):

- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `MLFLOW_POSTGRES_DB`
- `AIRFLOW__CORE__FERNET_KEY`, `_AIRFLOW_WWW_USER_*`
- `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`
- `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `S3_ARTIFACT_BUCKET`, `MINIO_MODELS_BUCKET`, `MINIO_REPORTS_BUCKET`, `MINIO_PREDICTIONS_BUCKET`
- `TARGET_COLUMN`, `ID_COLUMN`, `OPTUNA_N_TRIALS`, `PSI_THRESHOLD`, `PR_AUC_SHADOW_THRESHOLD`

`src/config.get_settings()` centraliza la lectura y propagación de estas variables.

## 9. Solución de Problemas

- **Errores de importación en Airflow**: verifica que el repositorio esté montado correctamente; los DAGs añaden `PROJECT_ROOT` al `PYTHONPATH` en tiempo de ejecución.
- **Falta LightGBM/XGBoost**: instala dependencias con `pip install -r requirements.txt` o incorpóralas en la imagen de Airflow.
- **Fallan los uploads a MinIO**: revisa credenciales `MINIO_*` y que los buckets existan (`mc alias ls`). Usa `docker compose run --rm minio-setup` si necesitas recrearlos.
- **Errores con el registry de MLflow**: asegúrate de que el servidor use Postgres como backend (ya configurado en el compose) y de que `MLFLOW_TRACKING_URI` apunte al servicio.
- **Fallo en exportar PDF con WeasyPrint**: requiere librerías del sistema (`pango`, `cairo`). Si no están, se registrará una advertencia y se omitirá el PDF.
- **No hay predicciones para monitoreo**: el DAG de monitoreo se salta la ejecución si no encuentra `data/predictions_*.csv`.

## 10. Próximos Pasos

1. Añadir CI para ejecutar pruebas unitarias y linting sobre `src/` y DAGs.
2. Extender `dag_predict_new_data` con ingesta desde DWH o APIs.
3. Conectar alertas (Slack/webhooks) para umbrales de PSI o performance.
4. Externalizar secretos y describir la infraestructura con IaC para mantenimiento a largo plazo.

---

Para revisar decisiones de feature engineering y supuestos del EDA, consulta `notebooks/churn_risk_eda_explainability.ipynb`.
