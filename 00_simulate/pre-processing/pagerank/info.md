# Bitcoin Centrality Analysis

This repository contains a two-step Spark pipeline for analyzing centrality in the Bitcoin transaction network.

## Project Layout

- `bitcoin_dat_decoder/`
  decodes raw Bitcoin `blk*.dat` files into a reusable edge list
- `bitcoin_pagerank/`
  reads the decoded edge list and computes PageRank with Spark RDDs
- `bitcoin_common/`
  shared helpers used by both jobs
- `docker-compose.yml`
  local Spark cluster used to run the pipeline in Docker

## Pipeline

1. Decode raw Bitcoin block files into CSV edges.
2. Aggregate and rank the resulting graph with PageRank.
3. Write results and run summaries under the project output folders.

## Local Run

The repository is designed to run inside the Spark Docker environment defined at the project root.

```bash
docker compose up -d
docker compose exec pyspark-client bash
```

Inside the container:

```bash
cd /opt/spark/work-dir/project/bitcoin_dat_decoder
SAMPLE_FILES=1 OVERWRITE=1 ./run_ingest_docker.sh

cd /opt/spark/work-dir/project/bitcoin_pagerank
MODE=basic ./run_docker.sh
```

## Notes

- The decoder produces reusable CSV edge lists for the ranking job.
- The PageRank job supports `basic`, `combined`, and `paper` modes.
- Results are written under `07_reports/pagerank/` and execution summaries are appended to `experiment_results/runs.csv`.
