"""
Dataset generation and balancing utilities for the Continuous Halting PoC.
"""
import itertools
import multiprocessing
import random
import time
import pandas as pd
from src.interpreter import inst_to_token, token_to_inst, execute_program

def get_all_instructions(N):
    """Returns all valid instructions for a program of length N."""
    insts = [
        ('HLT',),
        ('INC', 0), ('INC', 1),
        ('DEC', 0), ('DEC', 1)
    ]
    for target in range(N):
        insts.append(('JNZ', 0, target))
        insts.append(('JNZ', 1, target))
    return insts

def process_program(prog_tokens):
    """Worker function for parallel interpreter execution."""
    program = [token_to_inst(t) for t in prog_tokens if t != 0]
    halted, steps, trace = execute_program(program)
    
    trace_pc = [state[0] for state in trace]
    trace_r0 = [state[1] for state in trace]
    trace_r1 = [state[2] for state in trace]
    
    return {
        'program': list(prog_tokens),
        'length': len(program),
        'label': halted,
        'steps': steps,
        'trace_pc': trace_pc,
        'trace_r0': trace_r0,
        'trace_r1': trace_r1
    }

def generate_raw_programs(max_perm_len=4, sample_sizes=None):
    """
    Generates permutations and samples programs of length N <= 8.
    Pads programs with 0 (PAD) up to length 8.
    """
    if sample_sizes is None:
        sample_sizes = {N: 20000 for N in range(5, 9)}
        
    all_programs = []
    
    # 1. Complete permutations for small lengths (N <= max_perm_len)
    for N in range(1, max_perm_len + 1):
        insts = get_all_instructions(N)
        tokens = [inst_to_token(i) for i in insts]
        for prog in itertools.product(tokens, repeat=N):
            padded = list(prog) + [0] * (8 - N)
            all_programs.append(tuple(padded))
            
    # 2. Random sampling for larger lengths (max_perm_len < N <= 8)
    for N in range(max_perm_len + 1, 9):
        sample_size = sample_sizes.get(N, 20000)
        insts = get_all_instructions(N)
        tokens = [inst_to_token(i) for i in insts]
        
        max_possible = len(tokens) ** N
        actual_sample = min(sample_size, max_possible)
        
        sampled = set()
        while len(sampled) < actual_sample:
            prog = tuple(random.choices(tokens, k=N))
            padded = list(prog) + [0] * (8 - N)
            sampled.add(tuple(padded))
            
        all_programs.extend(sampled)
        
    # Deduplicate
    all_programs = list(set(all_programs))
    return all_programs

def balance_dataset(df):
    """Balances the dataset 50/50 for Halted (y=1) and Non-Halted (y=0) classes."""
    df_halted = df[df['label'] == 1]
    df_loop = df[df['label'] == 0]
    
    min_size = min(len(df_halted), len(df_loop))
    print(f"Original Counts: Halted (y=1) = {len(df_halted)}, Looping (y=0) = {len(df_loop)}")
    
    if min_size == 0:
        print("Warning: One of the classes is empty. Skipping balancing.")
        return df
        
    df_halted_balanced = df_halted.sample(n=min_size, random_state=42)
    df_loop_balanced = df_loop.sample(n=min_size, random_state=42)
    
    balanced_df = pd.concat([df_halted_balanced, df_loop_balanced]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Balanced Dataset Counts: Halted (y=1) = {min_size}, Looping (y=0) = {min_size}")
    return balanced_df

def run_dataset_generation(max_perm_len=4, sample_size=20000, output_path="halting_dataset.parquet"):
    """Runs the full generation, execution, balancing, and saving pipeline."""
    sample_sizes = {N: sample_size for N in range(max_perm_len + 1, 9)}
    
    print("Generating raw program permutations and samples...")
    raw_programs = generate_raw_programs(max_perm_len, sample_sizes)
    print(f"Total unique program inputs: {len(raw_programs)}")
    
    print("Executing programs in parallel...")
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(process_program, raw_programs)
    print(f"Executed in {time.time() - start_time:.2f} seconds.")
    
    df_raw = pd.DataFrame(results)
    
    print("\nDataset Statistics BEFORE Balancing:")
    print(df_raw.groupby('length')['label'].agg(['count', 'mean']))
    
    df_balanced = balance_dataset(df_raw)
    
    print("\nDataset Statistics AFTER Balancing:")
    print(df_balanced.groupby('length')['label'].agg(['count', 'mean']))
    
    print(f"\nSaving dataset to {output_path}...")
    df_balanced.to_parquet(output_path, index=False)
    print("Dataset generation complete!")
