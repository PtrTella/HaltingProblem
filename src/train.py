import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as transforms

from src.model import ContinuousHaltingModel


def plot_latent_space_pca(z_matrix, labels):
    """Riduce z(T) a 2D e plotta i cluster per TensorBoard"""
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_matrix.cpu().detach().numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        z_2d[:, 0], z_2d[:, 1], c=labels.cpu().numpy(), cmap="bwr", alpha=0.6
    )
    ax.set_title("Latent Space z(T) PCA Projection")
    plt.colorbar(scatter, label="0 = Loop, 1 = Halt")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    tensor_img = transforms.ToTensor()(img)
    plt.close(fig)
    return tensor_img


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    # Parametri operativi (euler in locale, dopri5 sul cluster)
    solver_method = "euler" if device.type == "cpu" else "dopri5"

    df = pd.read_parquet("halting_dataset.parquet")

    # Preparazione tensori
    programs = torch.tensor(np.stack(df["program"].values), dtype=torch.long)
    labels = torch.tensor(df["label"].values, dtype=torch.float32)
    registers_init = torch.zeros((len(df), 2), dtype=torch.float32)

    dataset = TensorDataset(programs, registers_init, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = ContinuousHaltingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    t_span = torch.tensor([0.0, 1.0]).to(device)

    writer = SummaryWriter("runs/halting_poc")

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss, total_cos = 0, 0

        for batch_prog, batch_reg, batch_y in loader:
            batch_prog, batch_reg, batch_y = (
                batch_prog.to(device),
                batch_reg.to(device),
                batch_y.to(device),
            )

            optimizer.zero_grad()
            logits, _, cos_sim = model(
                batch_prog, batch_reg, t_span, method=solver_method
            )

            # Loss principale (BCE)
            loss_bce = F.binary_cross_entropy_with_logits(logits, batch_y)

            # Penalità per forzare l'allineamento del coseno sui programmi che si fermano (y=1)
            # Vogliamo cos_sim -> 1 per i programmi che terminano.
            loss_align = (batch_y * (1.0 - cos_sim)).mean()

            loss = loss_bce + 0.1 * loss_align
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cos += cos_sim.mean().item()

        avg_loss = total_loss / len(loader)
        avg_cos = total_cos / len(loader)
        print(
            f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Avg Cosine Align: {avg_cos:.4f}"
        )

        writer.add_scalar("Training/Loss", avg_loss, epoch)
        writer.add_scalar("Metrics/CosineAlignment", avg_cos, epoch)

        # Logging visivo dello spazio latente (1 batch di test)
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                sample_prog, sample_reg, sample_y = next(iter(loader))
                sample_prog, sample_reg = sample_prog.to(device), sample_reg.to(device)
                _, trajectory, _ = model(
                    sample_prog, sample_reg, t_span, method=solver_method
                )
                img = plot_latent_space_pca(trajectory[-1], sample_y)
                writer.add_image("Latent_PCA", img, epoch)

    writer.close()
    print("Training completato. Esegui 'tensorboard --logdir=runs' per visualizzare.")


if __name__ == "__main__":
    train()
