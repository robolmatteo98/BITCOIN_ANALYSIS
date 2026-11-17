# 03 — Change address

## Idea

Identifica quale output di una transazione è probabilmente il resto restituito
al mittente. Se un input appartiene a un cluster e un output è classificato
come resto, l'indirizzo di quell'output può essere aggiunto allo stesso
cluster.

## Calcoli previsti

Per ogni output candidato si combinano segnali come:

```text
new_address             = l'indirizzo non già visto prima
script_match            = stesso tipo di script degli input
value_relation          = output con importo minore rispetto al totale input
position_signal         = posizione tipica nel wallet osservato
address_reuse_history   = indirizzo già controllato o già visto
```

Un punteggio possibile è una somma pesata normalizzata:

```text
P(change | output) = sigmoid(w1*new_address + w2*script_match
                              + w3*value_relation + w4*position_signal)
```

Il metodo richiede cautela: CoinJoin, batching e pagamenti con più destinatari
rendono ambigua la distinzione tra pagamento e resto.

