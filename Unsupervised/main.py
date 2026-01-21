from get_data.data_loading import load_data
from utility.graph_building import build_graph
from utility.model_vgae import train_vgae
from utility.anomaly_detection import detect_anomalies_latent
from utility.anomaly_detection_normalized import detect_anomalies_latent_normalized
from utility.anomaly_detection_Kmeans import detect_anomalies_cluster_distance
from classification.Anomaly_classification_with_scores import classify_node_with_scores
from utility.Reporting import save_anomaly_report
from utility.print_output import print_results
from utility.visualize_latent import visualize_latent_space
from utility.explain_outlier import explain_outlier

import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

df_edges, addresses = load_data()

print(df_edges)

data, addr_to_idx, node_stats = build_graph(df_edges, addresses, use_relative_time=False)

model, z = train_vgae(data)

indices_sospetti, scores, norms, threshold = detect_anomalies_latent(z)
#indices_sospetti, scores, norms, threshold = detect_anomalies_latent_normalized(z)
#indices_sospetti, scores, norms, threshold = detect_anomalies_cluster_distance(z)

print_results(indices_sospetti, df_edges, addresses)

# Salvataggio report
save_anomaly_report(
    addresses=addresses,
    norms=norms,
    indices_sospetti=indices_sospetti,
    df_edges=df_edges,
    classify_fn=classify_node_with_scores,
    threshold=threshold,
    num_nodes=len(addresses),
    output_path="./WEBAPP/backend/data"
)

"""visualize_latent_space(z, indices_sospetti, addresses, output_path="latent_space.png")

for idx in indices_sospetti:
    explain_outlier(idx, node_stats, addresses)"""