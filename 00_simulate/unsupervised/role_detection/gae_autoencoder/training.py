"""Training unsupervised del GAE per ottenere embedding dei wallet."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EPOCHS, HIDDEN_DIM, LATENT_DIM, LEARNING_RATE
from .graph_builder import GraphData
from .GCN_model import GraphAutoencoder
from .GAT_model import GraphAttentionAutoencoder


def train_model(
    data: GraphData,
    architecture: str = "gcn",
) -> GraphAutoencoder | GraphAttentionAutoencoder:

    # shape[0] = righe, shape[1] = colonne
    if architecture == "gcn":
        model = GraphAutoencoder(data.X.shape[1], HIDDEN_DIM, LATENT_DIM)
    elif architecture == "gat":
        model = GraphAttentionAutoencoder(
            data.X.shape[1], HIDDEN_DIM, LATENT_DIM
        )
    else:
        raise ValueError("architecture deve essere 'gcn' oppure 'gat'")

    # Prepara l'oggetto che verrà richiamato da optimizer.step() che avrà il compito di
    # aggiornare i parametri apprendibili del modello dopo ogni calcolo del gradiente.
    # model.parameters include tutti i parametri dell'encoder e del decoder.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Binary Cross Entropy, viene creata la funzione di perdita.
    criterion = nn.BCEWithLogitsLoss()


    edge_index = torch.cat([data.positive_edges, data.negative_edges], dim=1)

    for epoch in range(EPOCHS):

        # imposta il modello in modalità training
        model.train()

        # azzera i gradienti calcolati nell'iterazione precedente
        optimizer.zero_grad()

        # esegue il forward del Graph Autoencoder e restituisce Z
        Z = model.encode(data.X, data.P)

        # esegue lo score e restituisce i logits
        logits = model.score(Z, edge_index)

        # calcola quanto le predizioni del modello sono lontane dalle label corrette
        loss = criterion(logits, data.labels)

        # calcola i gradienti
        loss.backward()

        # aggiorna i parametri tramite Adam
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoca {epoch + 1} | Loss: {loss.item():.6f}")
            
    return model
