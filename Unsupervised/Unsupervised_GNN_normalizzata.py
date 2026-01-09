# =====================================================================
# Questa unsupervised GNN calcola qual è la struttura NORMALE del grafo Bitcoin
# i nodi che non si ricostruiscono bene --> anomali

import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE, GINEConv

from Generatore_da_db import load_bitcoin_edges_from_db_without_warning
from Anomaly_classification import classify_suspicious_node, classify_node_with_scores

from Reporting import save_anomaly_report

# =====================================================================
# 1) CARICAMENTO DATI
# =====================================================================
df_edges, addresses = load_bitcoin_edges_from_db_without_warning()

# =====================================================================
# 2) NORMALIZZAZIONE FLOW_AMOUNT e TIME
# =====================================================================

# Normalizzazione del flow_amount tramite logaritmo
df_edges["flow_amount_log"] = np.log1p(df_edges["flow_amount"])

# CAPIRE QUALE DELLE DUE NORMALIZZAZIONI è MEGLIO USARE
# 1. Normalizzazione del tempo tramite (min-max)
t_min = df_edges["time"].min()
t_max = df_edges["time"].max()
df_edges["time_norm"] = (df_edges["time"] - t_min) / (t_max - t_min)

# 2. Normalizzazione temporale relativa per nodo sorgente
grp = df_edges.groupby("from_address")["time"]
df_edges["time_rel"] = (df_edges["time"] - grp.transform("min")) / (grp.transform("max") - grp.transform("min") + 1e-9)

# =====================================================================
# 3) MAPPA INDIRIZZI → NODE INDEX
# =====================================================================

# Creazione mappatura indirizzi → indice
addr_to_idx = {addr: i for i, addr in enumerate(addresses)}

# Mappatura colonne 'from' e 'to'
df_edges[["src", "dst"]] = df_edges[["from_address", "to_address"]].applymap(addr_to_idx.get)

# Assicurati che 'time' sia numerico
df_edges["time"] = pd.to_numeric(df_edges["time"], errors="coerce")

num_nodes = len(addresses)

print(f"Num nodes: {num_nodes}")
print(f"Num edges: {len(df_edges)}")
print(df_edges.head())

# =====================================================================
# 4) COSTRUZIONE GRAPH DATA (PyG)
# =====================================================================
edge_index = torch.tensor(
  np.vstack([df_edges["src"].values, df_edges["dst"].values]),
  dtype=torch.long
)

# edge features: [flow_amount_log, time]
edge_attr = torch.tensor(
  df_edges[["flow_amount_log", "time_norm"]].values,
  dtype=torch.float
)

# node features: indegree + outdegree
in_degree = np.zeros(num_nodes)
out_degree = np.zeros(num_nodes)

for s, d in zip(df_edges["src"], df_edges["dst"]):
  out_degree[s] += 1
  in_degree[d] += 1

x = torch.tensor(np.vstack([in_degree, out_degree]).T, dtype=torch.float)

data = Data(
  x=x,
  edge_index=edge_index,
  edge_attr=edge_attr
)

edge_dim = edge_attr.shape[1]  # = 2

# =====================================================================
# 5) VGAE MODELLO
# =====================================================================
class Encoder(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = GINEConv(torch.nn.Sequential(
      torch.nn.Linear(in_channels, 64),
      torch.nn.ReLU(),
      torch.nn.Linear(64, 64)
    ))
    self.conv_mu = GCNConv(64, out_channels)
    self.conv_logstd = GCNConv(64, out_channels)

model = VGAE(Encoder(data.num_features, 16))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
  model.train()
  optimizer.zero_grad()
  z = model.encode(data.x, data.edge_index, data.edge_attr)
  loss = model.recon_loss(z, data.edge_index)
  loss += (1 / data.num_nodes) * model.kl_loss()
  loss.backward()
  optimizer.step()
  return loss.item()

print("Inizio training VGAE…")
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# ========================================================================================
#
# ========================================================================================

model.eval()
z = model.encode(data.x, data.edge_index, data.edge_attr).detach()

norms = torch.norm(z - z.mean(dim=0), dim=1)

# METODO MAD (robusto agli outlier)
median = norms.median()
mad = torch.median(torch.abs(norms - median))

threshold = median + 3 * mad
mask = norms > threshold

indices_sospetti = mask.nonzero(as_tuple=True)[0].tolist()
indirizzi_sospetti = [addresses[i] for i in indices_sospetti]

anomalies = [
  (addresses[i], norms[i].item())
  for i in indices_sospetti
]

# ordina per score decrescente
anomalies.sort(key=lambda x: x[1], reverse=True)

print("\n=== POTENZIALI INDIRIZZI SOSPETTI ===")
for indirizzo, norma in anomalies:
  print(f"{indirizzo} - score di anomalia: {norma:.4f}")


# =====================================================================
# 6) ANALISI DETTAGLIATA NODI SOSPETTI
# =====================================================================

print("\n=== DETTAGLI NODI SOSPETTI ===")
for i in indices_sospetti:
    addr = addresses[i]
    reason = classify_node_with_scores(addr, df_edges)
    node_edges = df_edges[(df_edges['from_address'] == addr) | (df_edges['to_address'] == addr)]
    out_edges = node_edges[node_edges['from_address'] == addr]
    in_edges = node_edges[node_edges['to_address'] == addr]
    
    print(f"\nIndirizzo sospetto: {addr}")
    for label, score in reason:
      print(f"  - {label}: {(score * 100):.2f} %")
    print(f"  Grado uscente: {len(out_edges)}, somma uscente: {out_edges['flow_amount'].sum():.2f}")
    print(f"  Grado entrante: {len(in_edges)}, somma entrante: {in_edges['flow_amount'].sum():.2f}")
    print(node_edges[['txid', 'from_address','to_address','flow_amount','time']])
    print("---------------------------------------------------")

"""report_path = save_anomaly_report(
  addresses=addresses,
  norms=norms,
  indices_sospetti=indices_sospetti,
  df_edges=df_edges,
  classify_fn=classify_node_with_scores,
  threshold=threshold,
  num_nodes=num_nodes,
  output_path="./WEBAPP/backend/data"
)

print(f"\n✔ Report salvato in {report_path}")"""

"""plot_graph_with_anomalies(
    df_edges=df_edges,
    suspicious_indices=indices_sospetti,
    addr_to_idx=addr_to_idx,
    filename="bitcoin_graph.png"
)"""