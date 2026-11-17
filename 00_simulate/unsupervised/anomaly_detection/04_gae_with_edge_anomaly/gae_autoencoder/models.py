"""Encoder GNN e decoder per singole transazioni."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, edge_dim):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.We1 = nn.Parameter(torch.randn(edge_dim, hidden_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, latent_dim) * 0.1)
        self.We2 = nn.Parameter(torch.randn(edge_dim, latent_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(latent_dim))

    @staticmethod
    def aggregate(num_nodes, destinations, messages):
        """Aggrega i messaggi ricevuti da ogni nodo destinazione."""
        result = torch.zeros(num_nodes, messages.shape[1], device=messages.device)
        result.index_add_(0, destinations, messages)
        degree = torch.ones(num_nodes, device=messages.device)
        degree.index_add_(0, destinations, torch.ones_like(destinations, dtype=torch.float32))
        return result / degree.unsqueeze(1)

    @staticmethod
    def aggregate_with_self(self_messages, destinations, incoming_messages):
        """Aggrega i messaggi dei vicini insieme al self-loop del nodo."""
        result = self_messages.clone()
        result.index_add_(0, destinations, incoming_messages)
        degree = torch.ones(self_messages.shape[0], device=self_messages.device)
        degree.index_add_(0, destinations, torch.ones_like(destinations, dtype=torch.float32))
        return result / degree.unsqueeze(1)

    def forward(self, X, edge_index, edge_features):
        sources, destinations = edge_index

        # Primo livello: il nodo riceve sia la propria informazione sia
        # un messaggio dai nodi sorgente collegati e dalle feature degli archi.
        node_messages_1 = X[sources] @ self.W1
        edge_messages_1 = edge_features @ self.We1
        H1 = self.aggregate_with_self(X @ self.W1, destinations, node_messages_1)
        H1 = H1 + self.aggregate(X.shape[0], destinations, edge_messages_1) + self.b1
        H1 = F.relu(H1)

        # Secondo livello: stessa propagazione, partendo dalla rappresentazione
        # intermedia H1 invece che dalle feature iniziali X.
        node_messages_2 = H1[sources] @ self.W2
        edge_messages_2 = edge_features @ self.We2
        H2 = self.aggregate_with_self(H1 @ self.W2, destinations, node_messages_2)
        H2 = H2 + self.aggregate(H1.shape[0], destinations, edge_messages_2) + self.b2
        return F.relu(H2)

class EdgeAwareDecoder(nn.Module):
    def __init__(self, latent_dim, edge_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)
        self.edge_weight = nn.Parameter(torch.randn(edge_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, Z, edge_index, edge_features):
        sources, destinations = edge_index
        structural = torch.sum((Z[sources] @ self.W) * Z[destinations], dim=1)
        return structural + edge_features @ self.edge_weight + self.bias

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, edge_dim):
        super().__init__()
        self.encoder = EdgeAwareGCNEncoder(input_dim, hidden_dim, latent_dim, edge_dim)
        self.decoder = EdgeAwareDecoder(latent_dim, edge_dim)

    def encode(self, X, edge_index, edge_features):
        return self.encoder(X, edge_index, edge_features)

    def score(self, Z, edge_index, edge_features):
        return self.decoder(Z, edge_index, edge_features)
