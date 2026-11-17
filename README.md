```bash
PYTHONPATH=00_simulate/unsupervised \
python3 -m role_detection_v1.run_temporal \
  --database bitcoin_gnn \
  --history-limit 1000 \
  --limit 500
```
