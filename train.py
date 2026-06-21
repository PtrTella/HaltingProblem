#!/usr/bin/env python
import argparse
from src.train import train

def main():
    parser = argparse.ArgumentParser(description="Continuous Halting PoC: Train Wrapper")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 100)"
    )
    args = parser.parse_args()
    
    train(epochs=args.epochs)

if __name__ == "__main__":
    main()
