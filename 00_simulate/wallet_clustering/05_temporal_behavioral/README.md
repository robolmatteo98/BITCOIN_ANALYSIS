# 05 — Temporal and behavioral similarity

## Idea

Confronta il comportamento degli indirizzi nel tempo. È pensato soprattutto
come metodo di supporto o come ranking di candidati, non come regola unica di
appartenenza.

## Calcoli previsti

Per ogni indirizzo si costruisce un vettore di feature, ad esempio:

```text
activity_count       = numero di transazioni
active_days          = giorni distinti di attività
active_span          = last_seen - first_seen
hour_histogram       = distribuzione delle ore UTC
amount_histogram     = distribuzione degli importi logaritmici
counterparty_histogram = distribuzione dei destinatari/mittenti
```

I vettori vengono normalizzati e confrontati con cosine similarity, distanza
di Jensen-Shannon o distanza euclidea. Un grafo di similarità può poi essere
clusterizzato con una soglia, DBSCAN o comunità pesate.

La stagionalità, i fusi orari e l'attività automatizzata devono essere gestiti
prima di interpretare il risultato come evidenza di controllo comune.
