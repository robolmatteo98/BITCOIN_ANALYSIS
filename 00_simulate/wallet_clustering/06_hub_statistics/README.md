# 06 — Hub statistics

Questa cartella analizza gli indirizzi che potrebbero comportarsi come hub,
senza modificarli o rimuoverli ancora dal clustering.

## Statistiche calcolate

Per ogni `source` vengono calcolati:

```text
input_tx_count       numero di tx_id distinti in cui compare
input_row_count      numero di righe/input osservati
unique_counterparty  numero di source distinti con cui compare
co_spending_events   numero di righe di co-spending con altri source
first_seen           primo timestamp osservato
last_seen            ultimo timestamp osservato
active_days          numero di giorni distinti di attività
```

Per ogni transazione vengono inoltre calcolati:

```text
input_count          numero di source della transazione
```

Non viene ancora applicata nessuna soglia e non viene scritto alcun CSV. Lo
scopo è osservare la distribuzione reale e decidere successivamente come
gestire hub, consolidamenti e transazioni anomale.

## Esecuzione

```bash
python3 06_hub_statistics/run.py
```
