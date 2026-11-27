# Analisi dei cluster e assegnazione dei ruoli

## Distinzione fondamentale

La pipeline produce tre risultati distinti:

1. il GCN/GAE costruisce gli embedding dei nodi;
2. K-Means raggruppa gli embedding;
3. una funzione separata interpreta i cluster con etichette descrittive.

Le etichette `collector-like`, `distributor-like`, `hub-like` e `user-like`
non sono classi osservate né ground truth. Sono interpretazioni euristiche dei
profili medi dei cluster.

## Vecchia interpretazione

La versione precedente sommava tre feature entranti e tre uscenti e usava una
soglia fissa del 15%:

```text
collector-like    se incoming > outgoing * 1.15
distributor-like  se outgoing > incoming * 1.15
hub-like          se l'attività totale era massima
user-like         altrimenti
```

Questa regola spiegava perché `user-like` poteva non comparire: era una classe
residuale. Inoltre la soglia era arbitraria e la decisione non considerava
esplicitamente saldo, flusso netto, durata o PageRank.

## Nuova interpretazione

La pipeline calcola, per ogni cluster, indicatori dalle medie delle feature
trasformate:

```text
incoming_pressure
  media di incoming_degree, incoming_counterparties e total_received

outgoing_pressure
  media di outgoing_degree, outgoing_counterparties e total_sent

directionality
  (incoming_pressure - outgoing_pressure)
  / (incoming_pressure + outgoing_pressure)

balance
  1 - abs(directionality)

hub_index
  combinazione di attività strutturale ed economica
```

Le etichette sono assegnate così:

```text
directionality >= +0.20  -> collector-like
directionality <= -0.20  -> distributor-like
cluster bilanciato ma con hub_index >= 90% del massimo -> hub-like
altrimenti               -> user-like
```

La soglia di `0.20` resta una scelta euristica, ma è simmetrica e interpretabile:
il ruolo direzionale richiede una prevalenza relativa significativa tra la
pressione entrante e quella uscente. La decisione e gli indicatori usati sono
salvati nel campo `roles[*].evidence` del report JSON.

## Interpretazione Bitcoin

Queste etichette non identificano automaticamente un exchange, un mining pool,
un tumbler o un utente reale. Per Bitcoin, un ruolo economico richiede spesso
anche feature temporali, distribuzione degli importi, frequenza, numero di
controparti, reciprocity, densità dell'ego-network e indirizzi già etichettati.

La letteratura sui network roles distingue inoltre i ruoli strutturali dalle
community: nodi lontani nel grafo possono avere lo stesso ruolo se condividono
un pattern di connessioni simile. Per questo il risultato va descritto come
`collector-like` o `distributor-like`, non come identificazione certa di un
tipo di entità.

## Validazione consigliata

Per rendere l'analisi difendibile è opportuno confrontare:

- silhouette e distribuzione delle dimensioni;
- stabilità degli assegnamenti cambiando seed;
- indicatori `evidence` dei cluster;
- indirizzi o wallet con etichetta esterna;
- feature importance di un classificatore supervisionato, quando esistono
  etichette affidabili.

La silhouette valuta compattezza e separazione, ma non dimostra che il nome del
ruolo sia semanticamente corretto. Il ruolo finale dovrebbe quindi essere
interpretato insieme alle feature e non dal solo loss della GNN.

## Riferimenti

- Henderson et al., [RolX: Structural Role Extraction & Mining in Large Graphs](https://research.google/pubs/rolx-structural-role-extraction-mining-in-large-graphs/).
- Rousseeuw, [Silhouettes: A graphical aid to the interpretation and validation of cluster analysis](https://doi.org/10.1016/0377-0427(87)90125-7).
- Zheng et al., [Characterizing key agents in the cryptocurrency economy through blockchain transaction analysis](https://link.springer.com/article/10.1140/epjds/s13688-021-00276-9).
- Dehshali e Bahrak, [Behavioral classification of Bitcoin addresses based on transaction history](https://shayunak.github.io/assets/Address_classification.pdf).
