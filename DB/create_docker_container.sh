#!/bin/bash

# Percorso del file .env (modificalo se necessario)
ENV_FILE="../.env"

# Carica le variabili dal .env
export $(grep -v '^#' $ENV_FILE | xargs)

# Percorso del dump da ripristinare
DUMP_FILE="./backups/dump_30000_utf8.sql"

# Nome del container
CONTAINER_NAME="bitcoin_postgres"

echo "Avvio container PostgreSQL..."

# Avvia il container PostgreSQL
docker run -d \
  --name $CONTAINER_NAME \
  -e POSTGRES_DB=$DB_NAME \
  -e POSTGRES_USER=$DB_USER \
  -e POSTGRES_PASSWORD=$DB_PASSWORD \
  -p $DB_PORT:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:16

echo "Aspetto che PostgreSQL sia pronto..."
sleep 5

echo "Ripristino del dump: $DUMP_FILE"

docker cp "$DUMP_FILE" $CONTAINER_NAME:/dump.sql

docker exec -u postgres $CONTAINER_NAME \
  psql -d $DB_NAME -f /dump.sql

echo "Restore completato!"
echo "Container PostgreSQL in esecuzione su: $DB_HOST:$DB_PORT"