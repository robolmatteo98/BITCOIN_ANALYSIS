import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE

from Graph import plot_graph_with_anomalies
from Anomaly_classification import classify_suspicious_node

import pandas as pd
import numpy as np
import random
import string
import time

# ================================
# 1) GENERA 30 indirizzi "bitcoin-like"
# ================================
def fake_btc_address():
    prefix = "bc1q"
    body = ''.join(random.choices(string.ascii_lowercase + string.digits, k=30))
    return prefix + body

addresses = [fake_btc_address() for _ in range(30)]


# ================================
# 2) GENERA 50 TRANSAZIONI REALISTICHE
# ================================
rows = []

def add_tx(src, dst, amount=None, timestamp=None):
    if amount is None:
        amount = np.round(np.random.uniform(0.001, 2.5), 8)
    if timestamp is None:
        timestamp = np.random.randint(1_600_000_000, 1_700_000_000)
    rows.append([src, dst, amount, timestamp])

# NORMAL TRANSACTIONS
for _ in range(35):
    src = random.choice(addresses)
    dst = random.choice(addresses)
    while src == dst:
        dst = random.choice(addresses)
    add_tx(src, dst)

# ================================
# 3) PATTERN SOSPETTI INTRODOTTI APPOSTA
# ================================

## (A) PEELING CHAIN: A → B → C → D → E
chain_nodes = random.sample(addresses, 5)
for i in range(4):
    add_tx(chain_nodes[i], chain_nodes[i+1], amount=np.round(1.0 - i * 0.15, 8))

## (B) HUB SPRAY: un indirizzo paga 6 indirizzi
hub = random.choice(addresses)
spray_targets = random.sample(addresses, 6)
for dst in spray_targets:
    if dst != hub:
        add_tx(hub, dst, amount=np.round(np.random.uniform(0.05, 0.5), 8))

## (C) SELF-SHUFFLE / MIXER-LIKE: A → B, B → A, A → C
mixA, mixB, mixC = random.sample(addresses, 3)
add_tx(mixA, mixB, 0.34567890)
add_tx(mixB, mixA, 0.34210000)
add_tx(mixA, mixC, 0.50000000)

# ================================
# 4) COSTRUISCI DATAFRAME
# ================================
df_edges = pd.DataFrame(rows, columns=["from_address", "to_address", "amount", "timestamp"])

df_edges = df_edges.rename(columns={
    "amount": "flow_amount",
    "timestamp": "time"
})

print(df_edges)
print("\nNumero totale di transazioni:", len(df_edges))
print("Numero indirizzi:", len(addresses))


# =====================================================================
# 1) GENERAZIONE GRAFO FAKE
# =====================================================================

"""np.random.seed(42)
addresses = [f"addr{i}" for i in range(30)]

rows = []
for _ in range(100):
    src = np.random.choice(addresses)
    dst = np.random.choice(addresses)
    while dst == src:
        dst = np.random.choice(addresses)
    amount = np.random.uniform(1, 10)
    time = np.random.randint(1_600_000_000, 1_650_000_000)
    rows.append([src, dst, amount, time])

df_edges = pd.DataFrame(rows, columns=['from_address','to_address','flow_amount','time'])"""

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

x = torch.tensor(np.vstack([in_degree, out_degree]).T, dtype=torch.float)

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