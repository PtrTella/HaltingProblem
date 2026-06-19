#!/usr/bin/env python
import argparse
from src.dataset import run_dataset_generation

def main():
    parser = argparse.ArgumentParser(description="Continuous Halting PoC: Dataset Generator")
    parser.add_argument(
        "--max-perm-len",
        type=int,
        default=4,
        help="Maximum program length to generate all permutations for (default: 4)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Number of random programs to sample for each length N > max-perm-len (default: 20000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="halting_dataset.parquet",
        help="Filename for the generated Parquet file (default: halting_dataset.parquet)"
    )
    args = parser.parse_args()
    
    run_dataset_generation(
        max_perm_len=args.max_perm_len,
        sample_size=args.sample_size,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
