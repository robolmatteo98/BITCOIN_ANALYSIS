import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np

def build_graph_features(df):
    # mapping address → indice
    addresses = pd.concat([df.from_address, df.to_address]).unique()
    addr_to_idx = {addr: i for i, addr in enumerate(addresses)}

    # edge index
    src = df.from_address.map(addr_to_idx).values
    dst = df.to_address.map(addr_to_idx).values

    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    #edge_index = torch.tensor([src, dst], dtype=torch.long) lento

    # feature semplici (placeholder, puoi migliorarle dopo)
    x = torch.randn(len(addresses), 16)

    # tempo sugli edge
    edge_time = torch.tensor(df.time.values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.edge_time = edge_time

    return data

def build_graph_features_for_anomaly(df):
    addresses = pd.concat([df.from_address, df.to_address]).unique()
    addr_to_idx = {addr: i for i, addr in enumerate(addresses)}
    idx_to_address = {i: addr for i, addr in enumerate(addresses)}

    src = df.from_address.map(addr_to_idx).values
    dst = df.to_address.map(addr_to_idx).values

    import numpy as np
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    x = torch.randn(len(addresses), 16)

    edge_time = torch.tensor(df.time.values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.edge_time = edge_time

    # mapping per anomaly detection
    data.idx_to_address = idx_to_address

    return data