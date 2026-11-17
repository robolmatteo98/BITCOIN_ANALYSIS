# 04 — Transaction graph

## Idea

Rappresenta il ledger come un grafo bipartito con nodi indirizzo e
transazione. Gli archi rappresentano input e output. Il clustering viene
calcolato sul grafo completo, non soltanto sulla proiezione degli input.

## Calcoli previsti

Dal grafo bipartito si possono costruire:

```text
W_address[a,b] = numero di transazioni condivise da a e b
Jaccard(a,b)   = |N(a) ∩ N(b)| / |N(a) ∪ N(b)|
```

Successivamente si applicano componenti connesse, Louvain, Leiden o clustering
gerarchico sui pesi. Le feature possono includere grado, volume, numero di
transazioni, vicinato e direzione degli archi.

Rispetto al co-spending è più ricco, ma anche più costoso e più sensibile a
servizi centralizzati che collegano moltissimi indirizzi.

