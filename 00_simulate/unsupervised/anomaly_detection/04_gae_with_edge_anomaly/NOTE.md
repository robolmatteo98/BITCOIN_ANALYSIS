# Note sull'anomaly detection temporale

## Importi fuori scala

Nel test corrente, le transazioni `A -> B` da 50 e 100 BTC non vengono necessariamente segnalate come anomale. Il modello considera contemporaneamente la struttura del grafo e le feature apprese, quindi un importo elevato può risultare compatibile se la relazione tra i wallet è già conosciuta.

Per rilevare meglio importi fuori scala sarà utile aggiungere feature relative alla storia del wallet, ad esempio:

- rapporto tra importo corrente e media storica del wallet;
- rapporto tra importo corrente e mediana storica;
- deviazione rispetto agli importi precedenti;
- percentile dell'importo rispetto alla distribuzione storica;
- numero di transazioni precedenti tra gli stessi wallet.

## Split temporale

Il modello viene addestrato sul passato e valuta solo transazioni successive al cutoff temporale. Con il dataset sintetico attuale, usando `timestamp <= 6` si ottengono:

- 32 transazioni per il training;
- 10 transazioni future per lo scoring.

Le transazioni future non devono entrare nel contesto del GNN prima del calcolo dello score. Dopo la valutazione possono essere aggiunte al grafo per simulare l'aggiornamento progressivo della rete.

Questo split offre più storia al modello, ma lascia soltanto 10 transazioni per il test: è adeguato per una simulazione iniziale, non per una valutazione statistica definitiva.
