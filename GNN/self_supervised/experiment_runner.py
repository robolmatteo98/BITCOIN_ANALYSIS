import random
from statistics import mean, pstdev

import numpy as np
import torch

from self_supervised.anomaly_detection.anomaly_detection_isolation_forest import detect_anomalies
from self_supervised.eval import evaluate_dot_product, evaluate_link_prediction_metrics
from self_supervised.train import train_model


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _aggregate_metric_values(run_results, metric_name):
    values = [result[metric_name] for result in run_results]
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def run_repeated_experiment(model_factory, data, seeds, top_k=10):
    run_results = []
    last_anomaly_scores = None

    for seed in seeds:
        set_global_seed(seed)
        model = model_factory()
        trained = train_model(model, data)

        dot_product_score = evaluate_dot_product(trained, data)
        link_metrics = evaluate_link_prediction_metrics(trained, data)

        last_anomaly_scores = detect_anomalies(trained, data, top_k=top_k)

        run_results.append(
            {
                "seed": seed,
                "dot_product": dot_product_score,
                "roc_auc": link_metrics["roc_auc"],
                "average_precision": link_metrics["average_precision"],
            }
        )

    return {
        "runs": run_results,
        "summary": {
            "dot_product": _aggregate_metric_values(run_results, "dot_product"),
            "roc_auc": _aggregate_metric_values(run_results, "roc_auc"),
            "average_precision": _aggregate_metric_values(run_results, "average_precision"),
        },
        "anomalies_last_run": last_anomaly_scores,
    }
