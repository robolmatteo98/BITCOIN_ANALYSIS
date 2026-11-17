"""Training del GAE edge-aware."""

import torch
import torch.nn as nn

from .config import EDGE_DIM, EPOCHS, HIDDEN_DIM, LATENT_DIM, LEARNING_RATE
from .data import GraphData
from .GCN_model import GraphAutoencoder as GCNGraphAutoencoder


def train_model(data: GraphData, model_class=GCNGraphAutoencoder, epochs: int = EPOCHS) -> nn.Module:
    # shape[0] = righe, shape[1] = colonne
    model = model_class(data.X.shape[1], HIDDEN_DIM, LATENT_DIM, EDGE_DIM)
    # optimizer.step() aggiorna tutti i parametri apprendibili del modello.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Binary Cross Entropy, viene creata la funzione di perdita.
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        # imposta il modello in modalità training
        model.train()
        # azzera i gradienti calcolati nell'iterazione precedente
        optimizer.zero_grad()
        # esegue il forward del Graph Autoencoder e restituisce Z, logits
        _, logits = model(data.X, data.P, data.edge_features)
        # contiene le coppie dei nodi usate per il training
        src, dst = data.train_pairs.T
        # calcola quanto le predizioni del modello sono lontane dalle label corrette
        loss = criterion(logits[src, dst], data.train_labels)
        # calcola i gradienti e aggiorna i parametri
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoca {epoch + 1} | Loss: {loss.item():.6f}")
    return model
