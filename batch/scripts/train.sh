#!/bin/bash
#BSUB -J lasernet
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=64GB]"
##BSUB -u s215805@dtu.dk
#BSUB -B 
#BSUB -N
#BSUB -o logs/lasernet_%J.out
#BSUB -e logs/lasernet_%J.err

# Load modules
module load python3/3.12.0
module load cuda/12.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Run the script
python train.py