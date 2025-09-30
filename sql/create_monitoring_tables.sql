CREATE SCHEMA IF NOT EXISTS monitoring;

CREATE TABLE IF NOT EXISTS monitoring.scoring_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(128) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP NOT NULL,
    rows_processed INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    error_rate DOUBLE PRECISION NOT NULL,
    notes JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring.segment_metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(128) NOT NULL,
    segment VARCHAR(128) NOT NULL,
    metric VARCHAR(128) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ejmplo de uso:
-- psql "$MONITORING_URI" -f sql/create_monitoring_tables.sql
