# ADDRESS clustering per capire se due o più indirizzi appartengono alla stessa identità:
# a cosa serve davvero la co-spending:

# 1. identificazione di wallet appartenenti alla stessa entità
# è probabile che appartengano allo stesso utente, o controllati dallo stesso wallet, oppure facciano parte della stessa infrastruttura (exchaange, servizio...)

# 2. ricostruzione del flusso di denaro
# è possibile seguire i movimenti di un'entità nel tempo, capire quanto possiede e come interagisce con le altre entità

# 3. riconoscimento di exchange, mixer, servizi
# hanno pattern riconoscibili

# 4. in combinazione con altre euristiche, permette di dare supporto a indagini forensi
## 4.1 ad esempio qua può essere utilizzata per derivare le varie nazionalità degli indirizzi associati a indirizzi che si conosce già la loro nazionalità

# 5. analisi economica della rete
# per capire la distribuzione della ricchezza e studiare il comportamento degli utenti

# CI SONO MOLTI limiti nell'analisi co-spending, per questo ho bisogno di altre EURISTICHE

import pandas as pd
import networkx as nx
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path=".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

print(f"Connecting to {DB_HOST}:{DB_PORT} as {DB_USER} to DB {DB_NAME}")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# QUERY: quali indirizzi sono stati usati come input nella stessa transazione
query = text("""
SELECT
    inp.fk_transaction_id AS txid,
    prev_out.fk_address_code AS address
FROM tx_input inp
JOIN tx_output prev_out
  ON prev_out.fk_transaction_id = inp.prev_transaction_id
 AND prev_out.n = inp.prev_vout
""")

with engine.connect() as conn:
  df_inputs = pd.read_sql(query, conn)

  grouped = df_inputs.groupby("txid")["address"].apply(list)

  # Costruzione del grafo co-spending
  # es. input [A1, A2, A3], allora il grafo aggiunge gli edge:
  # A1 -- A2
  # A1 -- A3
  # A2 -- A3
  G = nx.Graph()

  for addresses in grouped:
    if len(addresses) > 1:
      for i in range(len(addresses)):
        for j in range(i + 1, len(addresses)):
          G.add_edge(addresses[i], addresses[j])

  # Estrazione clusters
  clusters = list(nx.connected_components(G))

  # Creo mappatura, dove ogni indirizzo prende un ID del cluster al quale appartiene
  address_to_cluster = {}

  for cluster_id, addresses in enumerate(clusters):
    for addr in addresses:
      address_to_cluster[addr] = cluster_id

  # OUTPUT
  df_clusters = pd.DataFrame(
    address_to_cluster.items(),
    columns=["address", "cluster_id"]
  )

  df_clusters.to_csv('test.csv', index=False) 
  print(f"File CSV salvato in: test.csv")