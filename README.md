Airflow plus MLOps Starter, Windows WSL friendly

1, Requisitos previos
   a, Windows 10 u 11 con WSL2 y Ubuntu activo
   b, Docker Desktop con integración WSL habilitada
   c, Ejecuta los comandos siempre desde Ubuntu, no desde PowerShell
   d, Asigna al menos 6 GB de RAM a Docker en Settings, Resources

2, Preparación del entorno
   a, Copia .env.example a .env y ajusta valores mínimos
      AIRFLOW_UID=50000, usuario interno para permisos en Linux
      _AIRFLOW_WWW_USER_USERNAME y _AIRFLOW_WWW_USER_PASSWORD para el login
      Mantén las credenciales de MinIO por defecto para desarrollo
   b, Crea la carpeta data y coloca dataset.csv y data_descriptions.csv dentro
      Ruta local, ./data en el proyecto

3, Primera puesta en marcha
   a, Inicializa Airflow
      make init
   b, Levanta toda la pila
      make up
   c, Crea el bucket en MinIO por si no existe
      make create-bucket
   d, Accesos
      Airflow en http://localhost:8080
      MLflow en http://localhost:5000
      MinIO Console en http://localhost:9001
      Usuario y clave en .env

4, Ejecutar tu primer pipeline
   a, En Airflow, activa
      churn_data_quality para validar datos
      churn_training_pipeline para preparar datos, entrenar y registrar a Staging
      daily_inference para generar predicciones demo
   b, Verifica en MLflow el nuevo experimento y el modelo registrado
   c, Revisa en data, predictions.csv

5, Estructura de carpetas
   airflow, dags con tres DAGs de ejemplo
   airflow, include con scripts y configuración YAML
   data, tus CSV viven aquí
   services, reservado para ampliaciones
   tests, espacio para pruebas unitarias

6, Extensiones recomendadas
   a, CI CD con GitHub Actions, lint y pruebas, y validación de DAGs
   b, Data Quality con Great Expectations más suites dedicadas y almacenamiento de resultados en S3
   c, Monitoreo de drift con Evidently, genera reportes HTML y publícalos
   d, Versionado de datos con DVC o LakeFS, remoto en MinIO
   e, Feature Store con Feast si lo necesitas, arranca local con SQLite y evoluciona

7, Solución de problemas frecuentes en WSL
   a, Permisos de archivos en logs de Airflow, usa AIRFLOW_UID=50000
   b, Rutas, ejecuta docker compose desde Ubuntu para evitar rutas de Windows
   c, Espacio y RAM, si ves OOM, baja el número de workers o sube memoria en Docker Desktop

8, Comandos útiles
   make logs para ver logs de servicios
   make airflow-bash para abrir una shell dentro del contenedor de Airflow
   docker compose down para apagar, docker system prune para limpiar recursos

9, Producción en siguientes iteraciones
   CeleryExecutor con Redis y workers autoscalables
   Separación de Postgres por servicio
   Autenticación SSO para Airflow
   Prometheus y Grafana para métricas, y alertas sobre fallos de DAGs
   Registry de modelos con aprobación manual para promoción a Production
