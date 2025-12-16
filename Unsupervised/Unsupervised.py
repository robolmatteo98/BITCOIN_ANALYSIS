import os
import pandas as pd
import numpy as np
import torch

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE

from Analisi_unsupervised import analyze_suspicious_nodes
from Graph import plot_graph_with_anomalies

# =============================================================================
# 1) LOAD ENVIRONMENT VARIABLES
# =============================================================================
load_dotenv(".env")

DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")

print("Loaded DB config:", DB_NAME, DB_USER, DB_HOST, DB_PORT)

# Postgres connection URI
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URI)

# =============================================================================
# 2) SQL QUERY FOR GRAPH RECONSTRUCTION
# =============================================================================

query_edges = """
SELECT
    from_address,
    to_address,
    flow_amount,
    time
FROM flows
WHERE from_address IS NOT NULL
  AND to_address IS NOT NULL
  AND from_address <> to_address
"""

print("Eseguo query PostgreSQL…")

with engine.connect() as conn:
    df_edges = pd.read_sql(text(query_edges), conn)

print("Query completata — righe:", len(df_edges))
print(df_edges.head())

# =============================================================================
# 3) MAP ADDRESSES → NUMERIC NODE IDs
# =============================================================================

# MAP ADDRESSES → NUMERIC NODE IDs
addresses = pd.concat([df_edges["from_address"], df_edges["to_address"]]).unique()
addr_to_idx = {addr: i for i, addr in enumerate(addresses)}
num_nodes = len(addresses)

df_edges["src"] = df_edges["from_address"].map(addr_to_idx)
df_edges["dst"] = df_edges["to_address"].map(addr_to_idx)

print("Nodi totali nel grafo:", num_nodes)

# =============================================================================
# 4) BUILD GRAPH FOR PYTORCH GEOMETRIC
# =============================================================================

# Edge index (2 × num_edges)
edge_index = torch.tensor(
    [df_edges["src"].values, df_edges["dst"].values],
    dtype=torch.long
)

# Edge attributes: amount + timestamp
df_edges["time"] = df_edges["time"].astype(float)

edge_attr = torch.tensor(
    df_edges[["flow_amount", "time"]].values,
    dtype=torch.float
)

# Node features: indegree + outdegree
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

print("Graph creato:", data)

# =============================================================================
# 5) UNSUPERVISED GNN — VARIATIONAL GRAPH AUTOENCODER
# =============================================================================

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv_mu = GCNConv(64, out_channels)
        self.conv_logstd = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

model = VGAE(Encoder(data.num_features, 16))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === TRAINING LOOP ============================================================

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

print("Inizio training VGAE…")
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# =============================================================================
# 6) ANOMALY SCORING
# =============================================================================

model.eval()
z = model.encode(data.x, data.edge_index).detach()

# distanza dal centroide della distribuzione degli embeddings
norms = torch.norm(z - z.mean(dim=0), dim=1)

# top 50 anomalous nodes
top_anomalies = norms.topk(50)
indices_sospetti = top_anomalies.indices.tolist()

indirizzi_sospetti = [addresses[i] for i in indices_sospetti]

print("\n=== POTENZIALI INDIRIZZI SOSPETTI (TOP 50) ===")
for a in indirizzi_sospetti:
    print(a)

# =============================================================================
# 7) PRINT GRAPH
# =============================================================================

"""plot_graph_with_anomalies(
    df_edges=df_edges,
    suspicious_indices=indices_sospetti,
    addr_to_idx=addr_to_idx,
    filename="bitcoin_graph.png"
)"""

# df_edges = dataframe delle transazioni
# addresses = lista degli indirizzi
# indices_sospetti = top 50 nodi sospetti
# addr_to_idx = mappa indirizzo -> indice
analyze_suspicious_nodes(df_edges, addresses, indices_sospetti, addr_to_idx)