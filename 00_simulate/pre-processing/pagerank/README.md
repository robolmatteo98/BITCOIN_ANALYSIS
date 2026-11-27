# PageRank preprocessing

Il job legge gli archi direttamente da PostgreSQL. Il decoder `.dat` non fa
più parte della pipeline.

## Esecuzione locale

Dal root del repository:

```bash
spark-submit 00_simulate/pre-processing/pagerank/bitcoin_pagerank/src/main.py \
  --wallet-mapping-dir 07_reports/wallet_clustering/entity_clustering \
  --history-limit 10000 \
  --limit 11000
```

Il job esegue due passate coerenti con il wallet clustering:

- `history`: usa `history/address_to_wallet.csv`;
- `analysis`: usa `analysis/address_to_wallet.csv`.

I risultati finiscono in:
`07_reports/pre-processing/pagerank/{history,analysis}`.

Per ogni modalità vengono salvati sia i primi 20 nodi sia tutti i punteggi:

```text
pagerank_basic_all.csv
pagerank_combined_all.csv
pagerank_paper_all.csv
```

I file completi hanno le colonne `node,score` e possono essere uniti in
seguito in un'unica tabella con tre colonne di punteggio.

Il merge dei tre PageRank wallet si esegue con:

```bash
python3 00_simulate/pre-processing/pagerank/merge_scores.py
```

Per il merge dei risultati address:

```bash
python3 00_simulate/pre-processing/pagerank/merge_scores.py \
  --root 07_reports/pre-processing/pagerank/address
```

Per il PageRank a livello di indirizzo si omette `--wallet-mapping-dir`.
Le credenziali PostgreSQL vengono lette dalla configurazione del progetto o
dagli argomenti/variabili d'ambiente già usati dal database loader.

Con lo script Docker il livello address si avvia così:

```bash
00_simulate/pre-processing/pagerank/bitcoin_pagerank/run_docker.sh \
  --address-level \
  --history-limit 10000 \
  --limit 11000
```

I risultati address vengono salvati separatamente in
`07_reports/pre-processing/pagerank/address/{history,analysis}`.

## Esecuzione Docker/Spark

```bash
00_simulate/pre-processing/pagerank/bitcoin_pagerank/run_docker.sh \
  --history-limit 10000 \
  --limit 11000
```

Il compose avvia un master, un worker e un client Spark. Il database viene
raggiunto tramite `host.docker.internal` di default; i parametri `DB_HOST`,
`DB_PORT`, `DB_NAME`, `DB_USER` e `DB_PASSWORD` possono essere sovrascritti.

La versione attuale carica gli archi nel driver prima di passarli a Spark: è
adatta alla prima integrazione e ai test su dataset limitati. Il passaggio a
lettura distribuita dal database sarà una fase successiva.
