"""Inizializzazione e training."""

import torch
import torch.nn as nn

from .config import EPOCHS, HIDDEN_DIM, LATENT_DIM, LEARNING_RATE
from .data import GraphData
from .GCN_model import GraphAutoencoder as GCNGraphAutoencoder


def train_model(
    data: GraphData,
    model_class=GCNGraphAutoencoder,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    hidden_dim: int = HIDDEN_DIM,
    latent_dim: int = LATENT_DIM,
) -> nn.Module:
    # shape[0] = righe, shape[1] = colonne
    model = model_class(data.X.shape[1], hidden_dim, latent_dim)

    # Prepara l'oggetto che verrà richiamato da optimizer.step() che avrà il compito di
    # aggiornare i parametri apprendibili del modello dopo ogni calcolo del gradiente.
    # model.parameters include tutti i parametri dell'encoder e del decoder.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Binary Cross Entropy, viene creata la funzione di perdita.
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        # imposta il modello in modalità training
        model.train()

        # azzera i gradienti calcolati nell'iterazione precedente
        optimizer.zero_grad()

        # esegue il forward del Graph Autoencoder e restituisce Z, logits
        _, logits = model(data.X, data.P)

        # contiene le coppie dei nodi usate per il training
        src, dst = data.train_pairs.T

        # calcola quanto le predizioni del modello sono lontane dalle label corrette
        loss = criterion(logits[src, dst], data.train_labels)

        # calcola i gradienti
        loss.backward()
        
        # aggiorna i parametri tramite Adam
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoca {epoch + 1} | Loss: {loss.item():.6f}")

    return model
