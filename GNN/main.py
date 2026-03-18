from data_loader import load_flows
from data_loader_views import load_flows_with_address_features
from self_supervised.feature_engineering.feature_engineering import (
  build_graph_features_for_anomaly as build_random_features_for_anomaly,
)
from self_supervised.feature_engineering.feature_engineering_address_stats import (
  build_graph_features_for_anomaly as build_address_stats_for_anomaly,
)
from self_supervised.feature_engineering.feature_engineering_db_views import (
  build_graph_features_from_views_for_anomaly,
)
from self_supervised.experiment_runner import run_repeated_experiment

from models.GCN import GCN
from models.GraphSAGE import GraphSAGE
from models.GAT import GAT

FEATURE_PIPELINES = {
  "random": build_random_features_for_anomaly,
  "address_stats": build_address_stats_for_anomaly,
  "db_views": build_graph_features_from_views_for_anomaly,
}

COMPATIBLE_PIPELINES = {
  "flows_only": {"random", "address_stats"},
  "db_views": {"db_views"},
}

def main():
  data_source_name = "db_views"
  feature_pipeline_name = "db_views"
  experiment_seeds = [42, 43, 44]
  top_k_anomalies = 10

  if feature_pipeline_name not in FEATURE_PIPELINES:
    raise ValueError(f"Pipeline non supportata: {feature_pipeline_name}")

  compatible_pipelines = COMPATIBLE_PIPELINES.get(data_source_name)
  if compatible_pipelines is None:
    raise ValueError(f"Sorgente dati non supportata: {data_source_name}")
  if feature_pipeline_name not in compatible_pipelines:
    raise ValueError(
      f"Combinazione non supportata: data_source={data_source_name}, "
      f"feature_pipeline={feature_pipeline_name}"
    )

  if data_source_name == "flows_only":
    df = load_flows()
    data = FEATURE_PIPELINES[feature_pipeline_name](df)
  elif data_source_name == "db_views":
    df_flows, df_address_features = load_flows_with_address_features()
    data = FEATURE_PIPELINES[feature_pipeline_name](df_flows, df_address_features)

  print(f"=== Data source: {data_source_name} ===")

  print(f"\n=== Feature pipeline: {feature_pipeline_name} ===")
  if hasattr(data, "feature_names"):
    print("Feature usate:")
    for feature_name in data.feature_names:
      print(f"- {feature_name}")
  print(f"\n=== Experiment seeds: {experiment_seeds} ===")

  model_factories = {
    "GCN": lambda: GCN(data.num_features, 32, 2),
    "GraphSAGE": lambda: GraphSAGE(data.num_features, 32, 2),
    "GAT": lambda: GAT(data.num_features, 32, 2),
  }

  results = {}

  for name, model_factory in model_factories.items():
    experiment_result = run_repeated_experiment(
      model_factory=model_factory,
      data=data,
      seeds=experiment_seeds,
      top_k=top_k_anomalies,
    )
    results[name] = experiment_result

    print("\n<=== Modello: " + name + " ===>")
    for run_result in experiment_result["runs"]:
      print(
        f"seed={run_result['seed']} | "
        f"dot_product={run_result['dot_product']:.4f} | "
        f"roc_auc={run_result['roc_auc']:.4f} | "
        f"average_precision={run_result['average_precision']:.4f}"
      )
    print("Top anomalie ultima run:")
    for addr, score in experiment_result["anomalies_last_run"]:
      print(f"{addr} → score: {score:.3f}")

  print("\n=== Confronto modelli ===")
  print(f"{'Model':<15} {'Dot Product':<25} {'ROC AUC':<25} {'Avg Precision':<25}")

  for name, metrics in results.items():
    summary = metrics["summary"]
    print(
      f"{name:<15} "
      f"{summary['dot_product']['mean']:.4f}±{summary['dot_product']['std']:.4f}    "
      f"{summary['roc_auc']['mean']:.4f}±{summary['roc_auc']['std']:.4f}    "
      f"{summary['average_precision']['mean']:.4f}±{summary['average_precision']['std']:.4f}"
    )

if __name__ == "__main__":
  main()
