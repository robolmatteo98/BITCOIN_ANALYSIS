import torch
from torch_geometric.nn import VGAE, GINEConv


class EdgeAwareEncoder(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, latent_dim):
        super().__init__()

        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
            ),
            edge_dim=edge_dim,
        )

        self.conv_mu = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(64, latent_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(latent_dim, latent_dim),
            ),
            edge_dim=edge_dim,
        )

        self.conv_logstd = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(64, latent_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(latent_dim, latent_dim),
            ),
            edge_dim=edge_dim,
        )

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.conv1(x, edge_index, edge_attr))
        mu = self.conv_mu(h, edge_index, edge_attr)
        logstd = torch.clamp(self.conv_logstd(h, edge_index, edge_attr), min=-10, max=10)
        return mu, logstd


def train_vgae(data, epochs=300, lr=0.005, latent_dim=8, beta=5.0):
    model = VGAE(
        EdgeAwareEncoder(
            in_channels=data.num_features,
            edge_dim=data.edge_attr.shape[1],
            latent_dim=latent_dim,
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z = model.encode(data.x, data.edge_index, data.edge_attr)
        loss = model.recon_loss(z, data.edge_index)
        loss += beta * model.kl_loss() / data.num_nodes

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss {loss:.4f}")

    model.eval()
    z = model.encode(data.x, data.edge_index, data.edge_attr).detach()
    return model, z