SHELL := /bin/bash

include .env

up:
	docker compose --env-file .env up -d --build

down:
	docker compose --env-file .env down

logs:
	docker compose --env-file .env logs -f --tail=100

init:
	docker compose --env-file .env up airflow-init --exit-code-from airflow-init

ps:
	docker compose --env-file .env ps

stop:
	docker compose --env-file .env stop

restart: down up

airflow-bash:
	docker compose --env-file .env exec airflow bash

mlflow-ui:
	@echo "Open http://localhost:5000"

minio-console:
	@echo "Open http://localhost:9001"

create-bucket:
	docker compose --env-file .env up minio-setup
