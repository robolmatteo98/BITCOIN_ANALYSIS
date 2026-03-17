import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def build_graph_features(df):
  # costruzione del grafo
  G = nx.from_pandas_edgelist(
    df,
    source="from_address",
    target="to_address",
    edge_attr="flow_amount",
    create_using=nx.DiGraph()
  )

  # aggiunta caratteristiche ad ogni nodo
  for node in G.nodes():
    in_deg = G.in_degree(node)
    out_deg = G.out_degree(node)

    G.nodes[node]["x"] = [in_deg, out_deg]

  # conversione in pytorch geometric, trasformando il grafo in un oggetto Data di pytorch
  data = from_networkx(G)

  # trasforma la lista delle caratteristiche dei nodi in un Tensore pytorch
  data.x = torch.tensor(
    [G.nodes[n]["x"] for n in G.nodes()],
    dtype=torch.float
  )

  return data