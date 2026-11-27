# CoinJoin detection

Questa cartella contiene l'euristica CoinJoin usata come filtro nella pipeline
di wallet clustering. Lo score combina numero di input/output e importi di
output ripetuti; non e un'etichetta certa.

Lo script legge il database e salva gli score in:

```text
07_reports/wallet_clustering/coinjoin/coinjoin_scores.csv
```

Esecuzione:

```bash
python3 00_simulate/wallet_clustering/coinjoin_detection/run.py --threshold 0.75
```
