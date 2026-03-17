from data_loader import load_flows
from self_supervised.feature_engineering import build_graph_features, build_graph_features_for_anomaly
from self_supervised.train import train_model, evaluate_model
from self_supervised.anomaly_detection_isolation_forest import detect_anomalies

from models.GCN import GCN
from models.GraphSAGE import GraphSAGE
from models.GAT import GAT

def main():
  # carica dati dal database trasformandoli in DataFrame
  df = load_flows()

  # costruisce grafo e dataset
  data = build_graph_features_for_anomaly(df)

  # media dei dot product
  models = {
    "GCN": GCN(data.num_features, 32, 2),
    "GraphSAGE": GraphSAGE(data.num_features, 32, 2),
    "GAT": GAT(data.num_features, 32, 2)
  }

  results = {}

  for name, model in models.items():

    trained = train_model(model, data)
    res = evaluate_model(trained, data)

    results[name] = res

    scores = detect_anomalies(trained, data, top_k=10)

    print("\n<=== Modello: " + name + " ===>")
    for addr, score in scores:
      print(f"{addr} → score: {score:.3f}")

  print("\n=== Confronto modelli (valore normalizzato dot product) ===")
  for name, score in results.items():
    print(f"{name}: {score:.4f}")

if __name__ == "__main__":
  main()