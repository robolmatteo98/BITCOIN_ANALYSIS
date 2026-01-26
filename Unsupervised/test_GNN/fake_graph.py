import torch
from torch_geometric.data import Data

def fake_build_graph(df, addresses):
    addr_to_idx = {a: i for i, a in enumerate(addresses)}

    edge_index = []
    edge_attr = []

    for _, row in df.iterrows():
        src = addr_to_idx[row.from_address]
        dst = addr_to_idx[row.to_address]

        edge_index.append([src, dst])
        edge_attr.append([
            row.flow_amount,
            row.n_inputs,
            row.n_outputs,
            row.total_amount
        ])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Node features: degree + total volume
    num_nodes = len(addresses)
    x = torch.zeros((num_nodes, 2))

    for src, dst in edge_index.t():
        x[src, 0] += 1
        x[dst, 0] += 1

    for _, row in df.iterrows():
        x[addr_to_idx[row.from_address], 1] += row.flow_amount

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data