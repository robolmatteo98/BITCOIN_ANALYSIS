from get_data.data_loading import load_data
from utility.graph_building import build_graph
from utility.model_vgae import train_vgae
from utility.anomaly_detection import detect_anomalies_latent
from utility.anomaly_detection_normalized import detect_anomalies_latent_normalized
from utility.anomaly_detection_Kmeans import detect_anomalies_cluster_distance
from classification.Anomaly_classification_with_scores import classify_node_with_scores
from print_results.Reporting import save_anomaly_report
from print_results.print_output import print_results
from print_results.visualize_latent import visualize_latent_space
from print_results.explain_outlier import explain_outlier
from test_GNN.fake_graph import fake_build_graph

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

# test
data = fake_build_graph(df_edges, addresses)
# vero db
#data, addr_to_idx, node_stats = build_graph(df_edges, addresses, use_relative_time=False)

print(data)
print("Node features:\n", data.x)
print("Edge attr:\n", data.edge_attr)

model, z = train_vgae(data)

indices_sospetti, norms, threshold = detect_anomalies_latent(z)
#indices_sospetti, scores, norms, threshold = detect_anomalies_latent_normalized(z)
#indices_sospetti, scores, norms, threshold = detect_anomalies_cluster_distance(z)

print("Anomalie rilevate:", indices_sospetti)
print("Indirizzi:", [addresses[i] for i in indices_sospetti])

#print_results(indices_sospetti, df_edges, addresses)

# Salvataggio report
"""save_anomaly_report(
    addresses=addresses,
    norms=norms,
    indices_sospetti=indices_sospetti,
    df_edges=df_edges,
    classify_fn=classify_node_with_scores,
    threshold=threshold,
    num_nodes=len(addresses),
    output_path="./WEBAPP/backend/data"
)"""

"""visualize_latent_space(z, indices_sospetti, addresses, output_path="latent_space.png")

for idx in indices_sospetti:
    explain_outlier(idx, node_stats, addresses)"""

visualize_latent_space(
    z,
    indices_sospetti,
    addresses,
    output_path="latent_fake.png"
)