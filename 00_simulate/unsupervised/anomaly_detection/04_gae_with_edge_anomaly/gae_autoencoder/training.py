"""Training sul passato con negativi usati solo come regolarizzazione."""
import torch
import torch.nn as nn
from .config import EPOCHS, HIDDEN_DIM, LATENT_DIM, LEARNING_RATE
from .data import GraphData
from .models import GraphAutoencoder

def train_model(data: GraphData, epochs: int = EPOCHS) -> GraphAutoencoder:
    # Il modello normale usa 2 edge feature; quello aggregato ne usa 4.
    edge_dim = data.train_edge_features.shape[1]
    model = GraphAutoencoder(data.X.shape[1], HIDDEN_DIM, LATENT_DIM, edge_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    negative_features = torch.zeros(
        len(data.negative_edge_index.T),
        data.train_edge_features.shape[1],
    )
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        Z = model.encode(data.X, data.train_edge_index, data.train_edge_features)
        positive_logits = model.score(Z, data.train_edge_index, data.train_edge_features)
        negative_logits = model.score(Z, data.negative_edge_index, negative_features)
        logits = torch.cat([positive_logits, negative_logits])
        labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoca {epoch + 1} | Loss: {loss.item():.6f}")
    return model
