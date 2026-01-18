import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

# Utilizziamo tutte le transazioni per calcolare le statistiche / caratteristiche di ogni nodo
# Ma non facciamo comparire nel nel grafo il COINBASE perché altrimenti il modello viene distorto da un SUPER-NODO

def build_graph(df_edges, addresses, use_relative_time=False):
  num_nodes = len(addresses)

  # Mappatura indirizzi → indici
  addr_to_idx = {addr: i for i, addr in enumerate(addresses)}

  # Mappa gli indirizzi reali
  df_edges["src"] = df_edges["from_address"].map(addr_to_idx)
  df_edges["dst"] = df_edges["to_address"].map(addr_to_idx)

  # ----------------------------------------------------------------------
  # 1) COSTRUZIONE DEL GRAFO (solo archi NON coinbase)
  # ----------------------------------------------------------------------
  # Gli archi coinbase hanno src = NaN → li escludiamo dal grafo
  df_graph_edges = df_edges[df_edges["src"].notna()].copy()
  df_graph_edges["src"] = df_graph_edges["src"].astype(int)
  df_graph_edges["dst"] = df_graph_edges["dst"].astype(int)

  # Edge index
  edge_index = torch.tensor(
    np.vstack([df_graph_edges["src"].values, df_graph_edges["dst"].values]),
    dtype=torch.long
  )

  # Edge features
  df_graph_edges["flow_ratio"] = df_graph_edges["flow_amount"] / (df_graph_edges["total_amount"] + 1e-9)
  df_graph_edges["log_flow_ratio"] = np.log1p(df_graph_edges["flow_ratio"])

  time_col = "time_rel" if use_relative_time else "time_norm"

  edge_attr = torch.tensor(
    df_graph_edges[["flow_amount_log", time_col, "log_flow_ratio"]].values.astype(np.float32),
    dtype=torch.float
  )

  edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

  # ----------------------------------------------------------------------
  # 2) NODE FEATURES (usano TUTTE le transazioni, incluse coinbase)
  # ----------------------------------------------------------------------
  node_stats = pd.DataFrame(index=np.arange(num_nodes))

  # Gradi (coinbase contribuisce solo all'in_degree)
  node_stats["in_degree"] = (df_edges.groupby("dst").size().reindex(node_stats.index, fill_value=0))
  node_stats["out_degree"] = (df_graph_edges.groupby("src").size().reindex(node_stats.index, fill_value=0))

  for col in ["in_degree", "out_degree"]:
    node_stats[f"log_{col}"] = np.log1p(node_stats[col])

  # Volumi economici
  node_stats["total_in"] = (df_edges.groupby("dst")["flow_amount"].sum().reindex(node_stats.index, fill_value=0))

  node_stats["total_out"] = (df_graph_edges.groupby("src")["flow_amount"].sum().reindex(node_stats.index, fill_value=0))

  for col in ["total_in", "total_out"]:
    node_stats[f"log_{col}"] = np.log1p(node_stats[col])

  # Complessità transazionale
  node_stats["avg_log_n_inputs"] = (df_edges.groupby("dst")["n_inputs"].mean().reindex(node_stats.index, fill_value=0))

  node_stats["avg_log_n_outputs"] = (df_graph_edges.groupby("src")["n_outputs"].mean().reindex(node_stats.index, fill_value=0))

  # Tensor finale
  x = torch.tensor(
    node_stats[
      [
        "log_in_degree",
        "log_out_degree",
        "log_total_in",
        "log_total_out",
        "avg_log_n_inputs",
        "avg_log_n_outputs",
      ]
    ].values.astype(np.float32),
    dtype=torch.float
  )

  # Standardizzazione
  x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

  data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

  return data, addr_to_idx, node_stats
