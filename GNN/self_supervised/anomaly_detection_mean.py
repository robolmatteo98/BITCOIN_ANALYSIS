import torch

def detect_anomalies(model, data, top_k=10):
    """
    Anomaly detection semplice basata sulla distanza dal centro degli embedding.
    Restituisce punteggio tra 0 (non sospetto) e 1 (molto sospetto).

    Args:
        model: modello self-supervised addestrato (GCN, GraphSAGE, GAT)
        data: oggetto PyG Data contenente x e edge_index
        top_k: numero di nodi più anomali da restituire

    Returns:
        top_nodes: lista degli indirizzi più sospetti (top_k)
        scores: torch.tensor dei punteggi di tutti i nodi (0=normale, 1=sospetto)
    """
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)  # embedding nodi [num_nodes, embedding_dim]

    # centro degli embedding
    center = z.mean(dim=0)

    # distanza euclidea nodo → centro
    dist = torch.norm(z - center, dim=1)  # [num_nodes]

    # normalizza tra 0 e 1
    scores = (dist - dist.min()) / (dist.max() - dist.min())

    # ordina i nodi dal più sospetto al meno sospetto
    anom_scores = [(idx, score.item()) for idx, score in enumerate(scores)]
    anom_scores.sort(key=lambda x: x[1], reverse=True)  # più alto = più sospetto

    # mapping indice → address
    idx_to_address = getattr(data, "idx_to_address", list(range(data.num_nodes)))

    top_nodes = [idx_to_address[idx] for idx, _ in anom_scores[:top_k]]

    return top_nodes, scores