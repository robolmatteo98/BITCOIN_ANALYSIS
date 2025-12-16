import psycopg2
import torch
from torch_geometric.data import Data
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

print(f"Connecting to {DB_HOST}:{DB_PORT} as {DB_USER} to DB {DB_NAME}")

# Connessione al DB
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
cursor = conn.cursor()

# -------------------------------
# 1. INPUTS: address → tx
# -------------------------------
cur.execute("""
    SELECT
        ti.fk_transaction_id AS spending_txid,
        to2.fk_address_code AS from_address,
        to2.amount AS amount_in
    FROM bitcoin_tx_input ti
    JOIN bitcoin_tx_output to2
        ON to2.fk_transaction_id = ti.prev_transaction_id
       AND to2.n = ti.prev_vout
    WHERE to2.fk_address_code IS NOT NULL;
""")
inputs = cur.fetchall()

# -------------------------------
# 2. OUTPUTS: tx → address
# -------------------------------
cur.execute("""
    SELECT
        to1.fk_transaction_id AS txid,
        to1.fk_address_code AS to_address,
        to1.amount AS amount_out
    FROM bitcoin_tx_output to1
    WHERE to1.fk_address_code IS NOT NULL;
""")
outputs = cur.fetchall()

# -------------------------------
# 3. Create node index
# -------------------------------
addresses = {row[1] for row in inputs} | {row[1] for row in outputs}
addresses = list(addresses)
address_to_idx = {a: i for i, a in enumerate(addresses)}

txids = {row[0] for row in inputs} | {row[0] for row in outputs}
txids = list(txids)
tx_offset = len(addresses)
tx_to_idx = {tx: tx_offset + i for i, tx in enumerate(txids)}

num_nodes = len(addresses) + len(txids)

# -------------------------------
# 4. Build edges
# -------------------------------
src = []
dst = []
edge_attr = []

# INPUT edges: address → tx
for txid, addr, value in inputs:
    src.append(address_to_idx[addr])
    dst.append(tx_to_idx[txid])
    edge_attr.append([float(value)])

# OUTPUT edges: tx → address
for txid, addr, value in outputs:
    src.append(tx_to_idx[txid])
    dst.append(address_to_idx[addr])
    edge_attr.append([float(value)])

edge_index = torch.tensor([src, dst], dtype=torch.long)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

# -------------------------------
# 5. Node features
# -------------------------------
x = np.zeros((num_nodes, 4))

address_in = np.zeros(len(addresses))
address_out = np.zeros(len(addresses))
address_in_n = np.zeros(len(addresses))
address_out_n = np.zeros(len(addresses))

for txid, addr, value in inputs:
    idx = address_to_idx[addr]
    address_out[idx] += float(value)
    address_out_n[idx] += 1

for txid, addr, value in outputs:
    idx = address_to_idx[addr]
    address_in[idx] += float(value)
    address_in_n[idx] += 1

# tx features
tx_in_count = np.zeros(len(txids))
tx_out_count = np.zeros(len(txids))
tx_total_value = np.zeros(len(txids))

for txid, addr, value in inputs:
    t = tx_to_idx[txid] - tx_offset
    tx_in_count[t] += 1
    tx_total_value[t] += float(value)

for txid, addr, value in outputs:
    t = tx_to_idx[txid] - tx_offset
    tx_out_count[t] += 1

# Fill matrix
for i in range(len(addresses)):
    x[i] = [address_in[i], address_out[i], address_in_n[i], address_out_n[i]]

for i in range(len(txids)):
    idx = tx_offset + i
    x[idx] = [tx_in_count[i], tx_out_count[i], tx_total_value[i], 0]

x = torch.tensor(x, dtype=torch.float)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr
)

print(data)

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# 1. DATA
# -------------------------
# Assumiamo che tu abbia già il Data object pronto come "data"
# Se no: importa il tuo script bipartito prima di questo
# from your_data_script import data, address_to_idx

# -------------------------
# 2. DEFINIZIONE MODELLO UNSUPERVISED
# -------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(in_channels=data.x.size(1), hidden_channels=64, out_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------
# 3. TRAINING LOOP UNSUPERVISED
# -------------------------
for epoch in range(201):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    loss = (z.norm(dim=1).mean())  # semplice self-supervised loss
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# Embeddings finali
embeddings = z.detach().numpy()
print("Embeddings shape:", embeddings.shape)

# -------------------------
# 4. CLUSTERING
# -------------------------
k = 5  # numero di cluster, puoi cambiare
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Salva clusters
df_clusters = pd.DataFrame({
    "node_id": range(len(data.x)),
    "cluster": clusters
})
df_clusters.to_csv("node_clusters.csv", index=False)
print("Clusters salvati in node_clusters.csv")

# Salva embeddings
np.save("node_embeddings.npy", embeddings)
print("Embeddings salvati in node_embeddings.npy")

# -------------------------
# 5. VISUALIZZAZIONE PCA
# -------------------------
num_address_nodes = len(address_to_idx)
address_embeddings = embeddings[:num_address_nodes]
address_clusters = clusters[:num_address_nodes]

pca = PCA(n_components=2)
coords = pca.fit_transform(address_embeddings)

plt.figure(figsize=(8,6))
scatter = plt.scatter(coords[:,0], coords[:,1], c=address_clusters, cmap='tab10', s=10)
plt.title("PCA 2D embeddings indirizzi Bitcoin")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()
