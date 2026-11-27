# Role detection v1

Questa cartella contiene la prima versione della role detection basata su
feature aggregate dei nodi estratte dal database PostgreSQL.

La baseline originale resta separata in:

```text
00_simulate/unsupervised/role_detection/base/
```

I modelli GCN/GAT e il training condiviso dalle due versioni sono mantenuti in
`role_detection/gae_autoencoder/`.

## Obiettivo v1

Costruire un grafo `address → address` usando i flussi Bitcoin e descrivere
ogni indirizzo con feature strutturali, economiche e temporali:

- grado entrante e uscente;
- numero di controparti entranti e uscenti;
- output ricevuti, spesi e non spesi;
- numero e importo delle ricompense coinbase ricevute;
- cluster esplicito `coinbase-like` per gli indirizzi che ricevono ricompense;
- totale ricevuto, totale inviato e flusso netto;
- saldo stimato;
- prima e ultima attività;
- durata dell'attività;
- giorni attivi;
- numero di attività.
- dimensione e confidenza del wallet cluster;
- indicatore di wallet singleton;
- PageRank basic, combined e paper del wallet;
- PageRank basic, combined e paper dell'address, quando disponibili.

La sorgente principale sarà:

- `flows_view` per il grafo e i flussi;
- `07_reports/wallet_clustering/entity_clustering/{history,analysis}` per il mapping wallet;
- `07_reports/pre-processing/geolocation/{history,analysis}` per la geolocation;
- `07_reports/pre-processing/pagerank/{history,analysis}` per i PageRank wallet;
- `07_reports/pre-processing/pagerank/address/{history,analysis}` per i PageRank address.

Le feature economiche e temporali vengono calcolate dalla funzione SQL
`address_feature_summary_for_transactions(bigint[])`, passando gli stessi
`txid` del campione. In questo modo non vengono usate statistiche globali che
potrebbero contenere dati successivi al periodo analizzato.

La funzione è definita in:

```text
02_database/views/address_feature_summary_for_transactions.sql
```

La GNN e la classificazione dei ruoli restano inizialmente compatibili con la
baseline, così il confronto tra la versione statica e la v1 rimane possibile.

## Modalità del grafo

La pipeline supporta tre rappresentazioni:

- `address`: grafo `address -> address`, senza feature wallet;
- `address_wallet`: grafo `address -> address` con feature derivate dai wallet;
- `wallet`: grafo `wallet -> wallet`, ottenuto aggregando flussi e feature degli address.

La modalità predefinita è `address_wallet_feature`. Il vecchio nome
`address_wallet` resta accettato come alias.

```bash
# address -> address
PYTHONPATH=00_simulate/unsupervised \
python3 -m role_detection.v1.run_temporal \
  --graph-mode address --history-limit 10000 --limit 11000

# address -> address con feature wallet
PYTHONPATH=00_simulate/unsupervised \
python3 -m role_detection.v1.run_temporal \
  --graph-mode address_wallet --history-limit 10000 --limit 11000

# wallet -> wallet
PYTHONPATH=00_simulate/unsupervised \
python3 -m role_detection.v1.run_temporal \
  --graph-mode wallet --history-limit 10000 --limit 11000
```

I report vengono salvati automaticamente in cartelle separate per modalità:

```text
07_reports/unsupervised/role_detection/v1/
├── address/{history,analysis}/
├── address_wallet_feature/{history,analysis}/
└── wallet/{history,analysis}/
```

Il report JSON viene scritto in entrambe le partizioni; i grafici della
finestra di analisi vengono scritti nella rispettiva cartella `analysis`.

Nel livello wallet gli address non presenti nel mapping vengono mantenuti come
wallet singleton, così non vengono eliminati automaticamente dal grafo.

## Esecuzione

Dalla root del progetto:

```bash
PYTHONPATH=00_simulate/unsupervised \
python3 -m role_detection.v1.run \
  --history-limit 10000 \
  --limit 11000
```

Il comando usa `--limit` come cutoff totale e `--history-limit` come dimensione
del contesto storico. Per esempio, `--history-limit 10000 --limit 11000` usa
le prime 10000 transazioni come storico e le successive 1000 come finestra di
analisi. Il grafo rappresenta lo stato della rete
fino alla fine della finestra, senza dati successivi. Le coinbase contribuiscono
alle feature dei destinatari, ma non sono archi address → address: i relativi
indirizzi vengono mantenuti come nodi isolati e separati nel cluster
`coinbase-like`. Stampa le feature
utilizzate, le statistiche medie dei cluster e un solo wallet rappresentativo
per ogni ruolo.

Il report con embedding, assegnamenti e controllo di ricostruzione viene
salvato di default in `07_reports/unsupervised/role_detection/v1/report.json`.

## Pipeline temporale

La modalità temporale si avvia con:

```bash
PYTHONPATH=00_simulate/unsupervised \
python3 -m role_detection.v1.run_temporal \
  --history-limit 10000 \
  --limit 11000
```

La modalità temporale prova automaticamente i valori `k=4,...,10` dopo il
training, calcola la silhouette sugli embedding della finestra di analisi e
seleziona il valore con il punteggio più alto. Il report salva sia il risultato
del valore selezionato sia il confronto completo in `experiment.cluster_sweep`.
Per forzare un solo numero di cluster si può usare, ad esempio,
`--clusters 6`.

In questa modalità il GAE viene addestrato esclusivamente sulle transazioni
storiche. I suoi pesi vengono poi applicati al grafo della finestra successiva,
su cui vengono calcolati embedding e cluster.

La modifica più delicata riguarda i nuovi indirizzi presenti solo nella finestra
di analisi. Per questi il modello deve poter calcolare un embedding anche se non
erano presenti durante il training storico.
