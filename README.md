# BITCOIN ANALYSIS

Progetto dedicato all’analisi della rete Bitcoin tramite GNN, grafi, algoritmi di clustering.

## 00_simulate

Contiene le simulazioni e gli esperimenti di analisi non supervisionata sulla rete Bitcoin.

### unsupervised

Raccoglie le pipeline che analizzano la rete senza utilizzare etichette predefinite.
I modelli cercano di individuare autonomamente gruppi, comportamenti anomali e possibili ruoli degli indirizzi.

PIPELINE:

1. recupero dati da db o statici da codice
2. modello i dati, stabilendo quali feature sono importanti
3. creo il grafo
4. eseguo il modello GAE
   1. TRAINING
      1. encode: GCN
      2. decode: Bilinear
   2. Encode finale con pesi allenati
   3. Embedding Z
5. Cluster per capire quali indirizzi sono simili
6. Analizzo i risultati assegnando etichette

#### anomaly_detection

Questa sezione riguarda l’identificazione di indirizzi o comportamenti anomali nella rete.
Le anomalie vengono individuate confrontando le caratteristiche e gli embedding dei nodi.
Modelli encoder implementati: GCN, GAT
Modelli decoder implementati: Bilinear

##### gae_blank

Il modello conosce soltanto la struttura della rete, è la versione più semplice.

##### gae_with_node

Il modello conosce la struttura della rete e informazioni sui nodi. In questo caso sono: n di transazioni in entrata e in uscita, quantità di BTC ricevuti e inviati.

##### gae_with_edge

Il modello conosce la struttura della rete, informazioni sui nodi e in questo caso aggiunge informazioni sugli archi che sono: l'importo e il timestamp della transazione.

##### gae_with_edge_anomaly

Il modello conosce la struttura della rete, informazioni sui nodi e sugli archi.
Il modello viene addestrato sulle transazioni passate e valuta quelle future.
L'anomalia al momento viene indicata come una relazione tra i due nodi che è inattesa rispetto allo "storico".

#### role_detection

Versione base per individuare automaticamente i possibili ruoli dei wallet nella rete Bitcoin.
Utilizza una GCN con Graph Autoencoder per creare embedding degli indirizzi, che vengono poi raggruppati tramite clustering.

#### role_detection_v1

Studio più approfondito sulle possibili feature da assegnare ai nodi.

Versione migliorata della role detection, basata su feature aggregate strutturali, economiche e temporali estratte dal database.
Considera anche flussi, saldo stimato, attività nel tempo e indirizzi coinbase-like, mantenendo compatibilità con la versione base.

#### role_detection_v2

### wallet_clustering

#### co-spending

Utilizza il principio del co-spending per collegare indirizzi utilizzati insieme nella stessa transazione.
Consente di individuare gruppi di indirizzi probabilmente controllati dalla stessa entità.

## 01_acquisition

Contiene gli strumenti per acquisire e importare i dati della blockchain Bitcoin.
I blocchi vengono elaborati e inseriti nel database per le analisi successive.

## 02_database

Raccoglie la struttura del database, le tabelle e le viste utilizzate dal progetto.
Conserva i dati relativi a blocchi, transazioni, input, output e indirizzi.

## PageRank

Questa sezione riguarda l’applicazione dell’algoritmo PageRank al grafo della rete Bitcoin.
Permette di stimare l’importanza relativa degli indirizzi in base ai collegamenti e ai flussi ricevuti.