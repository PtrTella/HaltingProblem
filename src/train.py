import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import os
import datetime
from PIL import Image
import torchvision.transforms as transforms


try:
    from src.model import ContinuousHaltingModel
except ImportError:
    from model import ContinuousHaltingModel


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


def train(epochs=100, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    # Parametri operativi (euler in locale, dopri5 sul cluster)
    solver_method = "euler" if device.type == "cpu" else "dopri5"
    batch_size = 1024 if device.type == "cuda" else 256

    print("Loading dataset and splitting 80/20 train/val...")
    df = pd.read_parquet("halting_dataset.parquet")

    # Split train/validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Preparazione tensori train
    train_programs = torch.tensor(
        np.stack(train_df["program"].values), dtype=torch.long
    )
    train_labels = torch.tensor(train_df["label"].values, dtype=torch.float32)
    train_regs = torch.zeros((len(train_df), 2), dtype=torch.float32)

    # Preparazione tensori validation
    val_programs = torch.tensor(np.stack(val_df["program"].values), dtype=torch.long)
    val_labels = torch.tensor(val_df["label"].values, dtype=torch.float32)
    val_regs = torch.zeros((len(val_df), 2), dtype=torch.float32)

    train_dataset = TensorDataset(train_programs, train_regs, train_labels)
    val_dataset = TensorDataset(val_programs, val_regs, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Aumento capacità (embed_dim=32, latent_dim=256) per sfruttare i 48GB di VRAM
    model = ContinuousHaltingModel(embed_dim=32, latent_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Cosine Annealing Learning Rate Scheduler per gestire la stiffness
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t_span = torch.tensor([0.0, 1.0]).to(device)
    
    # Genera un ID run univoco basato sulla data/ora corrente
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=run_dir)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # --- Fase di Training ---
        model.train()
        total_train_loss, total_train_cos = 0.0, 0.0

        for batch_prog, batch_reg, batch_y in train_loader:
            batch_prog, batch_reg, batch_y = (
                batch_prog.to(device),
                batch_reg.to(device),
                batch_y.to(device),
            )

            optimizer.zero_grad()
            logits, _, cos_sim = model(
                batch_prog, batch_reg, t_span, method=solver_method
            )

            loss_bce = F.binary_cross_entropy_with_logits(logits, batch_y)
            # Forza l'allineamento del coseno con moltiplicatore aumentato a 0.5 per traiettorie più rettilinee
            loss_align = (batch_y * (1.0 - cos_sim)).mean()
            loss = loss_bce + 0.5 * loss_align

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_cos += cos_sim.mean().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_cos = total_train_cos / len(train_loader)

        # --- Fase di Validazione ---
        model.eval()
        total_val_loss, total_val_cos = 0.0, 0.0

        with torch.no_grad():
            for batch_prog, batch_reg, batch_y in val_loader:
                batch_prog, batch_reg, batch_y = (
                    batch_prog.to(device),
                    batch_reg.to(device),
                    batch_y.to(device),
                )
                logits, _, cos_sim = model(
                    batch_prog, batch_reg, t_span, method=solver_method
                )
                loss_bce = F.binary_cross_entropy_with_logits(logits, batch_y)
                loss_align = (batch_y * (1.0 - cos_sim)).mean()
                loss = loss_bce + 0.5 * loss_align

                total_val_loss += loss.item()
                total_val_cos += cos_sim.mean().item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_cos = total_val_cos / len(val_loader)

        # Step dello scheduler
        scheduler.step()

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (Cos: {avg_train_cos:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (Cos: {avg_val_cos:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Log metriche su TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/Train_CosineAlignment", avg_train_cos, epoch)
        writer.add_scalar("Metrics/Val_CosineAlignment", avg_val_cos, epoch)
        writer.add_scalar("Params/LearningRate", scheduler.get_last_lr()[0], epoch)

        # Checkpointing del modello basato sulla validation loss ed Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Salva checkpoint specifico della run
            checkpoint_path = os.path.join(run_dir, "best_halting_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            
            print(f"--> Saved best model checkpoint with Val Loss {best_val_loss:.4f} to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
                break

        # Logging visivo dello spazio latente (1 batch di validation, ogni 5 epoche)
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_prog, sample_reg, sample_y = next(iter(val_loader))
                sample_prog, sample_reg = sample_prog.to(device), sample_reg.to(device)
                _, trajectory, _ = model(
                    sample_prog, sample_reg, t_span, method=solver_method
                )
                img = plot_latent_space_pca(trajectory[-1], sample_y)
                writer.add_image("Latent_PCA", img, epoch)

    writer.close()
    print(f"Training completato. Pesi migliori salvati in '{os.path.join(run_dir, 'best_halting_model.pth')}'.")
    print("Esegui 'tensorboard --logdir=runs' per visualizzare.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuous Halting PoC: Train")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Number of epochs to wait for improvement before early stopping (default: 15)",
    )
    args = parser.parse_args()
    train(epochs=args.epochs, patience=args.patience)
