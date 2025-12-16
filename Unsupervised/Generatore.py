import pandas as pd
import numpy as np
import random
import string
import uuid

# =========================================================
# CONFIG
# =========================================================
NUM_ADDRESSES = 20
NUM_TRANSACTIONS = 50
INITIAL_UTXOS = 30

random.seed(42)
np.random.seed(42)

# =========================================================
# HELPERS
# =========================================================
def fake_btc_address():
    return "bc1q" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=30))

def new_txid():
    return str(uuid.uuid4())[:12]

# =========================================================
# 1) CREA INDIRIZZI
# =========================================================
addresses = [fake_btc_address() for _ in range(NUM_ADDRESSES)]

# =========================================================
# 2) CREA UTXO INIZIALI (simula mining / faucet)
# =========================================================
utxos = []
for _ in range(INITIAL_UTXOS):
    utxos.append({
        "txid": new_txid(),
        "vout": 0,
        "address": random.choice(addresses),
        "amount": round(np.random.uniform(0.5, 3.0), 8),
        "spent": False
    })

# =========================================================
# 3) GENERA TRANSAZIONI REALISTICHE
# =========================================================
transactions = []
edges = []
current_time = 1_600_000_000

def select_utxos(address, min_amount):
    available = [u for u in utxos if (u["address"] == address and not u["spent"])]
    random.shuffle(available)
    selected = []
    total = 0
    for u in available:
        selected.append(u)
        total += u["amount"]
        if total >= min_amount:
            break
    return selected, total

for _ in range(NUM_TRANSACTIONS):
    sender = random.choice(addresses)
    receiver = random.choice([a for a in addresses if a != sender])
    amount = round(np.random.uniform(0.05, 1.5), 8)
    fee = 0.0001

    inputs, total_in = select_utxos(sender, amount + fee)
    if total_in < amount + fee:
        continue  # sender non ha fondi sufficienti

    txid = new_txid()
    current_time += random.randint(60, 600)

    # marca UTXO come spesi
    for u in inputs:
        u["spent"] = True

    # output principale
    outputs = [{
        "txid": txid,
        "vout": 0,
        "address": receiver,
        "amount": amount,
        "spent": False
    }]

    # change
    change = round(total_in - amount - fee, 8)
    if change > 0.00001:
        change_addr = fake_btc_address()
        outputs.append({
            "txid": txid,
            "vout": 1,
            "address": change_addr,
            "amount": change,
            "spent": False
        })
        addresses.append(change_addr)

    utxos.extend(outputs)

    # salva transazione
    transactions.append({
        "txid": txid,
        "inputs": [u["address"] for u in inputs],
        "outputs": [o["address"] for o in outputs],
        "amounts": [o["amount"] for o in outputs],
        "time": current_time
    })

    # crea edge list address → address
    for o in outputs:
        edges.append([
            sender,
            o["address"],
            o["amount"],
            current_time,
            txid
        ])

# =========================================================
# 4) DATAFRAME FINALI
# =========================================================
df_edges = pd.DataFrame(
    edges,
    columns=["from_address", "to_address", "flow_amount", "time", "txid"]
)

df_txs = pd.DataFrame(transactions)

df_utxos = pd.DataFrame(utxos)

# =========================================================
# 5) STAMPA
# =========================================================
print("\n=== EDGE LIST (GNN INPUT) ===")
print(df_edges)

print("\nNumero transazioni:", len(df_txs))
print("Numero indirizzi:", len(set(df_edges["from_address"]).union(df_edges["to_address"])))
print("UTXO totali:", len(df_utxos))

def generate_bitcoin_dataset():
  return df_edges, addresses