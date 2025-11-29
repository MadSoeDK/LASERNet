#!/bin/bash
#BSUB -J test
#BSUB -q gpua10
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -B 
#BSUB -N
#BSUB -u s250062@dtu.dk
#BSUB -R "span[hosts=1]"
#BSUB -o logs/test_%J.out
#BSUB -e logs/test_%J.err

# Load modules
module load python3/3.12.0
module load cuda/12.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Run the script
python test.py