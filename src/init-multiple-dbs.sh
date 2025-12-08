#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create MLflow database
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;

    -- Create fraud_detection database
    CREATE DATABASE fraud_detection;
    GRANT ALL PRIVILEGES ON DATABASE fraud_detection TO $POSTGRES_USER;

    -- Create user for mlflow
    CREATE USER mlflow WITH PASSWORD 'mlflow';
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

    \c mlflow
    GRANT ALL ON SCHEMA public TO mlflow;
    ALTER SCHEMA public OWNER TO mlflow; 
EOSQL

echo "✅ Databases created: airflow, mlflow, fraud_detection"