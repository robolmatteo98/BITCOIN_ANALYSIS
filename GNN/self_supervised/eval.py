import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling

from self_supervised.train import temporal_edge_split


def evaluate_dot_product(model, data):
    model.eval()

    cutoff = data.edge_time.median()
    _, test_edges = temporal_edge_split(data, cutoff)

    with torch.no_grad():
        z = model(data.x, data.edge_index)

        src = test_edges[0]
        dst = test_edges[1]
        score = F.sigmoid((z[src] * z[dst]).sum(dim=1))

        return score.mean().item()


def evaluate_link_prediction_metrics(model, data):
    model.eval()

    cutoff = data.edge_time.median()
    _, test_edges = temporal_edge_split(data, cutoff)

    if test_edges.numel() == 0:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
        }

    with torch.no_grad():
        z = model(data.x, data.edge_index)

        pos_src = test_edges[0]
        pos_dst = test_edges[1]
        pos_logits = (z[pos_src] * z[pos_dst]).sum(dim=1)

        neg_edges = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=test_edges.size(1),
            method="sparse",
        )

        neg_src = neg_edges[0]
        neg_dst = neg_edges[1]
        neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=1)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        scores = torch.sigmoid(logits).cpu().numpy()
        labels = torch.cat(
            [
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits),
            ],
            dim=0,
        ).cpu().numpy()

    return {
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
    }
