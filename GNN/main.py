from data_loader import load_flows
from feature_engineering import build_graph_features
from train import train_model, evaluate_model

from models.GCN import GCN
from models.GraphSAGE import GraphSAGE
from models.GAT import GAT

import torch

def main():
    # 1 carica dati dal database trasformandoli in DataFrame
    df = load_flows()

    # 2 costruisce grafo e dataset
    data = build_graph_features(df)

    # 3 modelli
    models = {
      "GCN": GCN(data.num_features, 32, 2),
      "GraphSAGE": GraphSAGE(data.num_features, 32, 2),
      "GAT": GAT(data.num_features, 32, 2)
    }

    results = {}

    for name, model in models.items():

      # 4. training
      trained = train_model(model, data)

      # 5. evaluation
      res = evaluate_model(trained, data)

      results[name] = res

    print(results)

if __name__ == "__main__":
    main()