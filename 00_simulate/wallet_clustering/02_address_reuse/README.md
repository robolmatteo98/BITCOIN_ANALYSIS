# 02 — Address reuse

## Idea

Misura quanto lo stesso indirizzo ricompare in transazioni diverse. Il
riutilizzo può indicare continuità operativa, ma è un segnale debole: un
indirizzo pubblico può essere condiviso, pubblicato da un servizio o usato da
più persone.

## Calcoli previsti

Per ogni indirizzo `a` si calcolano:

```text
tx_count(a)       = numero di transazioni distinte in cui compare
input_tx_count(a) = numero di transazioni in cui viene speso
output_tx_count(a)= numero di transazioni in cui riceve
first_seen(a)     = prima altezza/data osservata
last_seen(a)      = ultima altezza/data osservata
reuse_rate(a)     = (tx_count(a) - 1) / max(tx_count(a), 1)
```

Due indirizzi possono ricevere un punteggio se hanno pattern di riutilizzo
simili, ad esempio stessa finestra temporale o comparsa sistematica nelle
stesse transazioni. Il metodo non dovrebbe creare cluster deterministici
senza una soglia e una valutazione dei falsi positivi.

