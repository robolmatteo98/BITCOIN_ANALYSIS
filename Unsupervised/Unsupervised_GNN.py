import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE

from Graph import plot_graph_with_anomalies
from Anomaly_classification import classify_suspicious_node

#from Generatore import generate_bitcoin_dataset
from Generatore_da_db import load_bitcoin_edges_from_db

# =====================================================================
# 1) GENERAZIONE GRAFO FAKE
# =====================================================================

#df_edges, addresses = generate_bitcoin_dataset()

# =====================================================================
# 1) CARICAMENTO GRAFO REALE BITCOIN
# =====================================================================

df_edges, addresses = load_bitcoin_edges_from_db(limit=50000)

print("Edges:", len(df_edges))
print("Nodes:", len(addresses))
print(df_edges.head())

# =====================================================================
# 2) MAPPA INDIRIZZI → NODE INDEX
# =====================================================================

addr_to_idx = {addr: i for i, addr in enumerate(addresses)}
df_edges["src"] = df_edges["from_address"].map(addr_to_idx)
df_edges["dst"] = df_edges["to_address"].map(addr_to_idx)

num_nodes = len(addresses)

# =====================================================================
# 3) COSTRUZIONE GRAPH DATA (PyG)
# =====================================================================

edge_index = torch.tensor(
    [df_edges["src"].values, df_edges["dst"].values],
    dtype=torch.long
)

edge_attr = torch.tensor(
    df_edges[["flow_amount", "time"]].values,
    dtype=torch.float
)

# node features: indegree + outdegree
in_degree = np.zeros(num_nodes)
out_degree = np.zeros(num_nodes)

for s, d in zip(df_edges["src"], df_edges["dst"]):
    out_degree[s] += 1
    in_degree[d] += 1

total_in = np.zeros(num_nodes)
total_out = np.zeros(num_nodes)

for s, d, amt in zip(df_edges["src"], df_edges["dst"], df_edges["flow_amount"]):
    total_out[s] += amt
    total_in[d] += amt

x = torch.tensor(
    np.vstack([in_degree, out_degree, total_in, total_out]).T,
    dtype=torch.float
)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr
)

# =====================================================================
# 4) MODELLO GNN UNSUPERVISED (VGAE)
# =====================================================================

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
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

# =====================================================================
# 5) ANOMALY SCORING
# =====================================================================

model.eval()
z = model.encode(data.x, data.edge_index).detach()

norms = torch.norm(z - z.mean(dim=0), dim=1)
top_anomalies = norms.topk(5)   # 5 nodi sospetti
indices_sospetti = top_anomalies.indices.tolist()

indirizzi_sospetti = [addresses[i] for i in indices_sospetti]

print("\n=== POTENZIALI INDIRIZZI SOSPETTI ===")
for a in indirizzi_sospetti:
    print(a)

# =====================================================================
# 6) ANALISI DETTAGLIATA NODI SOSPETTI
# =====================================================================

print("\n=== DETTAGLI NODI SOSPETTI ===")
for i in indices_sospetti:
    addr = addresses[i]
    reason = classify_suspicious_node(addr, df_edges)
    node_edges = df_edges[(df_edges['from_address'] == addr) | (df_edges['to_address'] == addr)]
    out_edges = node_edges[node_edges['from_address'] == addr]
    in_edges = node_edges[node_edges['to_address'] == addr]
    
    print(f"\nIndirizzo sospetto: {addr} --> {reason}")
    print(f"  Grado uscente: {len(out_edges)}, somma uscente: {out_edges['flow_amount'].sum():.2f}")
    print(f"  Grado entrante: {len(in_edges)}, somma entrante: {in_edges['flow_amount'].sum():.2f}")
    print(node_edges[['from_address','to_address','flow_amount','time']])
    print("---------------------------------------------------")

# =====================================================================
# 7) (OPZIONALE) PLOT GRAFO
# =====================================================================

plot_graph_with_anomalies(
    df_edges=df_edges,
    suspicious_indices=indices_sospetti,
    addr_to_idx=addr_to_idx,
    filename="bitcoin_graph_example.png"
)