import torch
from sklearn.ensemble import IsolationForest

def detect_anomalies(model, data, top_k=10, random_state=42):
    """
    Calcola gli embedding dei nodi usando il modello self-supervised
    e applica IsolationForest per trovare i nodi anomali.
    Restituisce i top_k nodi più sospetti già ordinati:
        1 = nodo più sospetto
        0 = nodo meno sospetto

    Args:
        model: modello self-supervised addestrato (GCN, GraphSAGE, GAT)
        data: oggetto PyG Data contenente x e edge_index
        top_k: numero di nodi più anomali da restituire
        random_state: seed per IsolationForest

    Returns:
        top_nodes: lista dei top_k indirizzi più sospetti (ordinati dal più sospetto al meno sospetto)
        top_scores: torch.tensor dei punteggi dei top_k nodi
    """
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)  # embedding nodi [num_nodes, embedding_dim]

    # IsolationForest
    clf = IsolationForest(contamination=0.01, random_state=random_state)
    clf.fit(z.numpy())

    # decision_function: più basso = più anomalo
    raw_scores = clf.decision_function(z.numpy())

    # più basso → 1, più alto → 0
    inverted_scores = -raw_scores

    # normalizziamo tra 0 e 1
    normalized_scores = (inverted_scores - inverted_scores.min()) / (inverted_scores.max() - inverted_scores.min())
    normalized_scores = torch.tensor(normalized_scores, dtype=torch.float)

    # mapping indice → address
    idx_to_address = getattr(data, "idx_to_address", list(range(data.num_nodes)))

    # creiamo lista (address, score) e ordiniamo decrescente
    anom_scores = [(idx_to_address[idx], score.item()) for idx, score in enumerate(normalized_scores)]
    anom_scores.sort(key=lambda x: x[1], reverse=True)  # più alto = più sospetto

    # prendiamo solo i top_k
    scores = anom_scores[:top_k]

    return scores