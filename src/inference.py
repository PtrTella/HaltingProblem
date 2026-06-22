"""
Inference Microscope and Gas Estimator for the Continuous Halting PoC.
Parses user programs, executes ODE inference, and projects trajectory
on top of the global latent space PCA attractor landscape.
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

try:
    from src.model import ContinuousHaltingModel
    from src.interpreter import inst_to_token
except ImportError:
    from model import ContinuousHaltingModel
    from interpreter import inst_to_token


def parse_instruction_string(inst_str):
    """
    Parses a string instruction into a tuple.
    Example:
        "HLT" -> ('HLT',)
        "INC R0" -> ('INC', 0)
        "DEC R1" -> ('DEC', 1)
        "JNZ R0 4" -> ('JNZ', 0, 4)
    """
    parts = inst_str.strip().split()
    op = parts[0].upper()
    if op == "HLT":
        return ("HLT",)
    elif op == "INC":
        reg = int(parts[1].replace("R", ""))
        return ("INC", reg)
    elif op == "DEC":
        reg = int(parts[1].replace("R", ""))
        return ("DEC", reg)
    elif op == "JNZ":
        reg = int(parts[1].replace("R", ""))
        target = int(parts[2])
        return ("JNZ", reg, target)
    raise ValueError(f"Unknown instruction string syntax: {inst_str}")


def program_to_tokens(program_str_list, max_len=8):
    """Converts list of instruction strings to a padded/truncated token list of length max_len."""
    tokens = []
    for inst_str in program_str_list:
        inst_tuple = parse_instruction_string(inst_str)
        tokens.append(inst_to_token(inst_tuple))

    # Pad or truncate to max_len
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Continuous Halting PoC: Interactive Inference Microscope"
    )
    parser.add_argument(
        "program_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to a text file containing string instructions",
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=0.0,
        help="Initial value of register R0 (default: 0.0)",
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=0.0,
        help="Initial value of register R1 (default: 0.0)",
    )
    args = parser.parse_args()

    # Default user program: a clean infinite loop
    user_program = ["INC R0", "DEC R0", "JNZ R0 0", "HLT"]

    # Allow user to pass a program file
    if args.program_file and os.path.exists(args.program_file):
        print(f"Reading user program from file: {args.program_file}")
        with open(args.program_file, "r") as f:
            user_program = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    print("\n" + "=" * 50)
    print("USER PROGRAM:")
    print("\n".join(f"  Line {i}: {line}" for i, line in enumerate(user_program)))
    print(f"INITIAL REGISTERS: R0={args.r0}, R1={args.r1}")
    print("=" * 50)

    tokens = program_to_tokens(user_program, max_len=8)
    print(f"Token IDs (Length 8): {tokens}")

    # Load model on CPU for local microscope usage
    device = torch.device("cpu")
    model = ContinuousHaltingModel(embed_dim=32, latent_dim=256).to(device)

    checkpoint_paths = ["runs/best_halting_model.pth", "best_halting_model.pth"]
    loaded = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded model weights from: {path}")
            loaded = True
            break

    if not loaded:
        print("Warning: Trained weights not found. Running with random initialization.")

    model.eval()

    # --- PCA Attractor Landscape Setup ---
    dataset_path = "halting_dataset.parquet"
    pca = None
    ref_zT_2d = None
    ref_labels = None

    if os.path.exists(dataset_path):
        print(f"Loading reference programs from {dataset_path} to build landscape...")
        try:
            df_ref = pd.read_parquet(dataset_path)
            # Sample 500 reference programs to draw the attractor landscape
            df_ref = df_ref.sample(n=min(500, len(df_ref)), random_state=42)
            ref_progs = torch.tensor(
                np.stack(df_ref["program"].values), dtype=torch.long
            ).to(device)
            ref_regs = torch.zeros((len(df_ref), 2), dtype=torch.float32).to(device)
            ref_labels = df_ref["label"].values

            # Forward pass for reference final states z(T) at t=1.0
            t_span_ref = torch.tensor([0.0, 1.0]).to(device)
            with torch.no_grad():
                _, ref_trajectories, _ = model(
                    ref_progs, ref_regs, t_span_ref, method="dopri5"
                )
                ref_zT = ref_trajectories[-1].numpy()  # shape: (500, latent_dim)

            pca = PCA(n_components=2)
            ref_zT_2d = pca.fit_transform(ref_zT)
            print("Successfully mapped background landscape via PCA.")
        except Exception as e:
            print(
                f"Error building landscape: {e}. Falling back to trajectory-only PCA."
            )
            pca = None
    else:
        print(
            "Warning: halting_dataset.parquet not found. Visualizing trajectory without background landscape."
        )

    # --- Run Single Trajectory Inference ---
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    regs_tensor = torch.tensor([[args.r0, args.r1]], dtype=torch.float32).to(device)

    # Integrate over 100 steps to get a smooth trajectory
    t_span = torch.linspace(0.0, 1.0, 100).to(device)

    with torch.no_grad():
        logits, trajectory, cos_sim = model(
            tokens_tensor, regs_tensor, t_span, method="dopri5"
        )

    probability = torch.sigmoid(logits).item()
    prediction = "HALT" if probability > 0.5 else "LOOP"

    print("\n" + "=" * 50)
    print("INFERENCE RESULTS:")
    print(f"Halt Probability: {probability:.6f}")
    print(f"Predicted Class:  {prediction}")
    print(f"Cosine Alignment at t=1: {cos_sim[-1].item():.4f}")

    # Trajectory Arc Length (Gas metric)
    # trajectory shape: (100, 1, latent_dim)
    traj_points = trajectory.squeeze(1)  # shape: (100, latent_dim)
    diffs = traj_points[1:] - traj_points[:-1]
    norms = torch.norm(diffs, p=2, dim=-1)
    arc_length = norms.sum().item()
    print(f"Estimated Trajectory Arc Length (Gas): {arc_length:.6f}")
    print("=" * 50 + "\n")

    # --- Trajectory Visualization ---
    plt.figure(figsize=(10, 8))

    if pca is not None and ref_zT_2d is not None:
        # Plot background landscape
        colors = ["#ff4d4d" if label == 1 else "#1a75ff" for label in ref_labels]
        plt.scatter(
            ref_zT_2d[:, 0],
            ref_zT_2d[:, 1],
            c=colors,
            alpha=0.12,
            s=25,
            label="Landscape (Blue=Loop, Red=Halt)",
        )

        # Project trajectory onto pre-fitted PCA
        traj_2d = pca.transform(traj_points.numpy())
        plt.plot(
            traj_2d[:, 0],
            traj_2d[:, 1],
            color="#2eb82e",
            linewidth=3.0,
            label="Trajectory Path",
            zorder=10,
        )
        plt.scatter(
            traj_2d[0, 0],
            traj_2d[0, 1],
            color="#00ffff",
            s=150,
            edgecolors="black",
            linewidths=1.5,
            zorder=11,
            label="Start (t=0)",
        )
        plt.scatter(
            traj_2d[-1, 0],
            traj_2d[-1, 1],
            color="#ff00ff",
            s=150,
            edgecolors="black",
            linewidths=1.5,
            zorder=11,
            label="End (t=1)",
        )
    else:
        # Fallback PCA on trajectory points
        pca_fallback = PCA(n_components=2)
        traj_2d = pca_fallback.fit_transform(traj_points.numpy())
        plt.plot(
            traj_2d[:, 0],
            traj_2d[:, 1],
            color="#2eb82e",
            linewidth=3.0,
            label="Trajectory Path",
        )
        plt.scatter(
            traj_2d[0, 0], traj_2d[0, 1], color="#00ffff", s=150, label="Start (t=0)"
        )
        plt.scatter(
            traj_2d[-1, 0], traj_2d[-1, 1], color="#ff00ff", s=150, label="End (t=1)"
        )

    plt.title(
        f"Program Trajectory PCA (Halt Prob: {probability:.2%})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.4)

    output_img = (
        args.program_file.replace(".txt", "_trajectory.png")
        if args.program_file
        else "./test/trajectory_viz.png"
    )
    plt.savefig(output_img, bbox_inches="tight", dpi=150)
    print(f"Trajectory visualization saved successfully as: {output_img}")


if __name__ == "__main__":
    main()
