import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class LLVMGraphEncoder(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, latent_dim=256):
        super().__init__()
        # 3-layer GNN Conv to propagate topological information across basic blocks
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Final MLP projection to continuous latent state z(0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU()
        )

    def forward(self, x, edge_index, batch=None):
        """
        x: [num_nodes, input_dim]
        edge_index: [2, num_edges]
        batch: [num_nodes] (PyG batch index, defaults to all zeros for single graph)
        """
        # Handle single graph case where batch is None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Message passing layers
        h = self.conv1(x, edge_index)
        h = F.silu(h)
        h = self.conv2(h, edge_index)
        h = F.silu(h)
        h = self.conv3(h, edge_index)
        h = F.silu(h)
        
        # Global pooling (Stage 3: compress CFG to fixed size)
        pooled = global_mean_pool(h, batch)
        
        # Project to continuous latent dimension
        z0 = self.fc(pooled)
        return z0
