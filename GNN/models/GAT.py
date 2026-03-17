import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden)
        self.conv2 = GATConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x