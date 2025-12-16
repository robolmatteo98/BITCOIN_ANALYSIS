import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import psycopg2
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
cur = conn.cursor()

cur.execute(
  """
  SELECT
    from_address,
    to_address,
    SUM(flow_amount) AS total_btc_sent,
    COUNT(*) AS total_tx_sent
  FROM flows
  WHERE from_address IS NOT NULL
    AND to_address IS NOT NULL
    AND from_address <> to_address
  GROUP BY from_address, to_address
  """
)
rows_raw = cur.fetchall()

rows = [
    {
        'from_address': r[0],
        'to_address': r[1],
        'total_btc_sent': float(r[2]),
        'total_tx_sent': r[3]
    }
    for r in rows_raw
]


# Lista di tutti gli indirizzi unici
unique_addresses = list({r['from_address'] for r in rows}.union({r['to_address'] for r in rows}))
address_to_idx = {addr: i for i, addr in enumerate(unique_addresses)}
num_nodes = len(unique_addresses)

# -------------------------
# 2. EDGE INDEX e EDGE ATTR
# -------------------------
source = [address_to_idx[r['from_address']] for r in rows]
target = [address_to_idx[r['to_address']] for r in rows]
edge_index = torch.tensor([source, target], dtype=torch.long)

# edge features: BTC totale e numero di transazioni
edge_attr = torch.tensor([[r['total_btc_sent'], r['total_tx_sent']] for r in rows], dtype=torch.float)

# -------------------------
# 3. FEATURE DEI NODI
# -------------------------
# inizializza feature dei nodi: [BTC_in, BTC_out, tx_in, tx_out]
x = np.zeros((num_nodes, 4))
for r in rows:
    src = address_to_idx[r['from_address']]
    tgt = address_to_idx[r['to_address']]
    
    x[src, 1] += r['total_btc_sent']   # BTC out
    x[src, 3] += r['total_tx_sent']    # tx out
    
    x[tgt, 0] += r['total_btc_sent']   # BTC in
    x[tgt, 2] += r['total_tx_sent']    # tx in

x = torch.tensor(x, dtype=torch.float)

# -------------------------
# 4. LABEL DEI NODI
# -------------------------
# Esempio: 0=wallet, 1=exchange, 2=mixer
# Qui inserisci le etichette conosciute
labels_dict = {
}

y = torch.tensor([labels_dict.get(addr, -1) for addr in unique_addresses], dtype=torch.long)

# Maschere train/test
train_mask = y != -1
test_mask = y == -1

# -------------------------
# 5. CREAZIONE DATASET
# -------------------------
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
            train_mask=train_mask, test_mask=test_mask)

# -------------------------
# 6. DEFINIZIONE MODELLO GNN
# -------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=x.size(1), hidden_channels=64, out_channels=3)  # 3 classi

# -------------------------
# 7. TRAINING LOOP
# -------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# -------------------------
# 8. PREDIZIONI
# -------------------------
model.eval()
pred_class = model(data.x, data.edge_index).argmax(dim=1)
print("Predizioni nodi test:", pred_class[data.test_mask])
