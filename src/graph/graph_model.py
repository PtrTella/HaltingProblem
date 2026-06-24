import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from src.model import ODEFunc
from src.graph.encoder import LLVMGraphEncoder

class GraphContinuousHaltingModel(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, latent_dim=256):
        super().__init__()
        # GNN encoder to compress LLVM CFG to latent state z(0)
        self.encoder = LLVMGraphEncoder(input_dim, hidden_dim, latent_dim)
        
        # Neural ODE dynamics func (governing the continuous time vector field flow)
        self.ode_func = ODEFunc(latent_dim)
        
        # Classification projection (maps final state z(T) to Halt probability logits)
        self.project_out = nn.Linear(latent_dim, 1)
        
        # Normal vector defining the halting attractor topological manifold
        self.n_halt = nn.Parameter(torch.randn(latent_dim))

    def forward(self, data, t_span, method="euler"):
        """
        data: PyTorch Geometric Data object or batch representing the LLVM IR CFG(s)
        t_span: torch.Tensor time grid (e.g. [0.0, 1.0])
        method: ode solver name (e.g. "euler", "dopri5")
        """
        # 1. Extract features, edge index and batch mapping
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # 2. Encode to initial state z(0)
        z0 = self.encoder(x, edge_index, batch)
        batch_size = z0.size(0)
        
        # 3. Integrate over time span using Neural ODE
        trajectory = odeint(self.ode_func, z0, t_span, method=method)
        zT = trajectory[-1]  # state at final time T
        
        # 4. Predict Halt Probability
        logits = self.project_out(zT).squeeze(-1)
        
        # 5. Compute direction alignment (cosine similarity of derivative vs attractor normal vector)
        dz_dt = self.ode_func(t_span[-1], zT)
        cos_sim = F.cosine_similarity(
            dz_dt, self.n_halt.unsqueeze(0).expand(batch_size, -1), dim=-1
        )
        
        return logits, trajectory, cos_sim
