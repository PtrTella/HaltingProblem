import torch
from src.graph.llvm_parser import parse_llvm_ir, INSTRUCTION_VOCAB
from src.graph.graph_model import GraphContinuousHaltingModel

def main():
    print("====================================================")
    print("Testing Stage 2: LLVM IR Graph Pipeline PoC Integration")
    print("====================================================\n")
    
    # 1. Define dummy LLVM IR string (a standard conditional loop)
    dummy_llvm_ir = """
    define i32 @main(i32 %n) {
    entry:
      %x = alloca i32, align 4
      store i32 %n, i32* %x, align 4
      br label %loop

    loop:
      %val = load i32, i32* %x, align 4
      %cond = icmp slt i32 %val, 10
      br i1 %cond, label %body, label %exit

    body:
      %val2 = load i32, i32* %x, align 4
      %inc = add nsw i32 %val2, 1
      store i32 %inc, i32* %x, align 4
      br label %loop

    exit:
      ret i32 0
    }
    """
    
    print("1. Parsing LLVM IR string...")
    # 2. Parse LLVM IR
    data = parse_llvm_ir(dummy_llvm_ir)
    
    print("Parsed PyTorch Geometric Data object:")
    print(f"  x shape: {data.x.shape}")
    print(f"  edge_index:\n{data.edge_index}")
    print(f"  Number of nodes (Basic Blocks): {data.x.size(0)}")
    print(f"  Number of edges: {data.edge_index.size(1)}")
    
    # Verify shape is correct: 4 basic blocks, 28 feature dimensions
    assert data.x.shape == (4, len(INSTRUCTION_VOCAB) + 1), f"Unexpected node feature shape: {data.x.shape}"
    assert data.edge_index.shape == (2, 4), f"Unexpected edge index shape: {data.edge_index.shape}"
    
    print("\nParsed instruction counts per Basic Block:")
    block_names = ["entry", "loop", "body", "exit"]
    for idx, name in enumerate(block_names):
        counts = data.x[idx].tolist()
        non_zero_insts = {INSTRUCTION_VOCAB[i]: int(counts[i]) for i in range(len(INSTRUCTION_VOCAB)) if counts[i] > 0}
        if counts[-1] > 0:
            non_zero_insts['other'] = int(counts[-1])
        print(f"  Block '{name}': {non_zero_insts}")
        
    print("\n2. Instantiating GraphContinuousHaltingModel...")
    # 3. Instantiate model
    # GNN input dim = 28 (VOCAB + 1), hidden_dim = 128, latent_dim = 256
    model = GraphContinuousHaltingModel(input_dim=len(INSTRUCTION_VOCAB) + 1, hidden_dim=128, latent_dim=256)
    model.eval()
    
    print("\n3. Running forward pass through GNN + Neural ODE (dopri5 solver)...")
    # 4. Integrate over 100 points in t in [0, 1]
    t_span = torch.linspace(0.0, 1.0, 100)
    
    with torch.no_grad():
        logits, trajectory, cos_sim = model(data, t_span, method="dopri5")
        
    halt_prob = torch.sigmoid(logits).item()
    
    print("\n====================================================")
    print("RESULTS:")
    print("====================================================")
    print(f"z(0) Shape:          {trajectory[0].shape} (batch_size=1, latent_dim=256)")
    print(f"z(T) Shape:          {trajectory[-1].shape} (batch_size=1, latent_dim=256)")
    print(f"Trajectory Shape:    {trajectory.shape} (steps=100, batch_size=1, latent_dim=256)")
    print(f"Halt Probability:    {halt_prob:.6f} ({'HALT' if halt_prob > 0.5 else 'LOOP'})")
    print(f"Cosine Similarity:   {cos_sim.item():.6f} (normal vector alignment)")
    print("====================================================")
    
    # Calculate geometric trajectory arc length (Gas metric)
    traj_points = trajectory.squeeze(1) # (100, 256)
    diffs = traj_points[1:] - traj_points[:-1]
    norms = torch.norm(diffs, p=2, dim=-1)
    arc_length = norms.sum().item()
    print(f"Estimated Trajectory Arc Length (Gas): {arc_length:.6f}")
    
    print("\nSuccess! The entire GNN + Neural ODE pipeline flows correctly!")

if __name__ == '__main__':
    main()
