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

La sorgente principale sarà:

- `flows_view` per il grafo e i flussi;

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

## Esecuzione

Dalla root del progetto:

```bash
PYTHONPATH=00_simulate/unsupervised \
python3 -m role_detection.v1.run \
  --database bitcoin_gnn \
  --history-limit 1000 \
  --limit 500
```

Il comando usa le prime 1000 transazioni come contesto storico e le successive
500 come finestra finale del campione. Il grafo rappresenta lo stato della rete
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
  --database bitcoin_gnn \
  --history-limit 1000 \
  --limit 500
```

In questa modalità il GAE viene addestrato esclusivamente sulle transazioni
storiche. I suoi pesi vengono poi applicati al grafo della finestra successiva,
su cui vengono calcolati embedding e cluster.

La modifica più delicata riguarda i nuovi indirizzi presenti solo nella finestra
di analisi. Per questi il modello deve poter calcolare un embedding anche se non
erano presenti durante il training storico.
