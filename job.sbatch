#!/bin/bash
#SBATCH --mail-user=pietro.tellarini2@studio.unibo.it
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=31G
#SBATCH --partition=l40
#SBATCH --output=runs/training_halting.log
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/HaltingProblem
#SBATCH --gres=gpu:1

# Attiva l'ambiente
source .venv/bin/activate

# Esegue il training
python -m src.train