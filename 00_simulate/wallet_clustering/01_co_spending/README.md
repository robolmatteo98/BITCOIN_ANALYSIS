# 01 — Co-spending / Common-input ownership

## Idea

Se una transazione spende contemporaneamente più output appartenenti agli
indirizzi `A`, `B` e `C`, il metodo assume che chi controlla `A`, `B` e `C`
abbia potuto firmare la stessa transazione. Gli indirizzi vengono quindi
collegati.

Questa è un'euristica, non una certezza: CoinJoin, pagamenti collaborativi e
servizi di mixing possono produrre falsi collegamenti.

## Calcoli

Il dataset statico ha una riga per ogni input e contiene:

```text
tx_id, source, destination, amount, timestamp
```

Per il calcolo del clustering si usano `tx_id` e `source`. Gli altri campi
sono già presenti per le analisi successive. Nel database reale l'indirizzo
`source` si ottiene risolvendo l'output precedente tramite:

```text
tx_input.prev_transaction_id + tx_input.prev_vout
    -> tx_output.fk_address_id
    -> address.code
```

Per ogni transazione si ottiene quindi un insieme di indirizzi:

```text
tx_11 -> {A1, A2}
tx_12 -> {A2, A3}
tx_13 -> {B1, B2, B3}
```

Gli indirizzi di ogni transazione sono già disponibili nel `group` prodotto da
`groupby`. L'implementazione conserva una lista di insiemi e fonde due insiemi
quando condividono almeno un indirizzo. Non è quindi necessario creare tutti
gli archi tra tutte le coppie.

```text
A1 -- A2 -- A3       => {A1, A2, A3}
B1 -- B2 -- B3       => {B1, B2, B3}
```

L'unione è transitiva: se `A1` è nello stesso insieme di `A2` e `A2` nello
stesso insieme di `A3`, tutti e tre finiscono nello stesso cluster. Le
componenti con meno di `MIN_CLUSTER_SIZE` indirizzi vengono escluse; la
costante vale `2`.

## Complessità

Per il dataset piccolo questa soluzione è più semplice da leggere. Su un
database molto grande si potrebbe sostituire la lista di insiemi con una
struttura union-find per ridurre le ricerche tra cluster.

## Esecuzione

```bash
python3 run.py
```

Il comando usa direttamente il dataset statico in `example_data.py`: non
serve alcun database, non vengono effettuate query SQL e non viene scritto
alcun file di output. Il dataset contiene
11 righe, 10 indirizzi sorgente, 5 transazioni e i relativi nodi destinazione,
importi e timestamp.

Quando il metodo sarà validato, la stessa funzione `find_clusters` potrà
ricevere il dataframe estratto dal database reale.