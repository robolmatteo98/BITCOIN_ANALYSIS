#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
COMPOSE_FILE="${PROJECT_DIR}/00_simulate/pre-processing/pagerank/docker-compose.yml"
MAPPING_ARGS=(
  --wallet-mapping-dir
  /opt/spark/work-dir/project/07_reports/wallet_clustering/entity_clustering
)
OUTPUT_DIR=/opt/spark/work-dir/project/07_reports/pre-processing/pagerank

if [[ "${1:-}" == "--address-level" ]]; then
  MAPPING_ARGS=()
  OUTPUT_DIR=/opt/spark/work-dir/project/07_reports/pre-processing/pagerank/address
  shift
fi

docker compose -f "${COMPOSE_FILE}" down --remove-orphans >/dev/null 2>&1 || true
if ! docker network inspect pagerank_default >/dev/null 2>&1; then
  docker network create --driver bridge pagerank_default >/dev/null
fi

docker compose -f "${COMPOSE_FILE}" up -d --build spark-master spark-worker pyspark-client
docker compose -f "${COMPOSE_FILE}" exec pyspark-client \
  /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
  /opt/spark/work-dir/project/00_simulate/pre-processing/pagerank/bitcoin_pagerank/src/main.py \
  "${MAPPING_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
