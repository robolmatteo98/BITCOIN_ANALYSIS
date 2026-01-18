import numpy as np
import pandas as pd
from get_data.Generatore_da_db import load_bitcoin_edges_from_db_without_warning

def load_data():
  df_edges, addresses = load_bitcoin_edges_from_db_without_warning()

  # Normalizzazione flow_amount
  df_edges["flow_amount_log"] = np.log1p(df_edges["flow_amount"])

  # Normalizzazione tempo globale
  t_min, t_max = df_edges["time"].min(), df_edges["time"].max()
  df_edges["time_norm"] = (df_edges["time"] - t_min) / (t_max - t_min)

  # Normalizzazione tempo relativa per nodo sorgente
  grp = df_edges.groupby("from_address")["time"]
  df_edges["time_rel"] = (df_edges["time"] - grp.transform("min")) / (grp.transform("max") - grp.transform("min") + 1e-9)

  df_edges['total_amount'] = df_edges['total_amount'].fillna(1e-9) # Evita lo zero assoluto

  return df_edges, addresses