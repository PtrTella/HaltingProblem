import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Rete rigorosamente C1-continua (SiLU), mai ReLU per non bloccare l'integrazione
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, t, z):
        return self.net(z)


class ContinuousHaltingModel(nn.Module):
    def __init__(self, vocab_size=22, seq_len=8, embed_dim=16, latent_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Proiezione z(0): embedding del codice + 2 registri (R0, R1)
        self.project_in = nn.Sequential(
            nn.Linear(seq_len * embed_dim + 2, latent_dim), nn.SiLU()
        )

        self.ode_func = ODEFunc(latent_dim)
        self.project_out = nn.Linear(latent_dim, 1)

        # Vettore normale appreso che rappresenta la "varietà topologica di arresto"
        self.n_halt = nn.Parameter(torch.randn(latent_dim))

    def forward(self, code_tokens, registers, t_span, method="euler"):
        batch_size = code_tokens.size(0)

        # 1. Embedding iniziale
        embedded_code = self.embedding(code_tokens).view(batch_size, -1)

        # 2. Stato iniziale z(0)
        z0_input = torch.cat([embedded_code, registers], dim=-1)
        z0 = self.project_in(z0_input)

        # 3. Evoluzione continua tramite ODE
        trajectory = odeint(self.ode_func, z0, t_span, method=method)
        zT = trajectory[-1]  # Stato al tempo T

        # 4. Previsione finale
        logits = self.project_out(zT).squeeze(-1)

        # 5. Calcolo direzione di evoluzione (Coseno tra derivata e vettore di arresto)
        dz_dt = self.ode_func(t_span[-1], zT)
        cos_sim = F.cosine_similarity(
            dz_dt, self.n_halt.unsqueeze(0).expand(batch_size, -1), dim=-1
        )

        return logits, trajectory, cos_sim
