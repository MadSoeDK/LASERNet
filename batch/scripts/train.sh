#!/bin/bash
#BSUB -J lasernet
#BSUB -q gpua40
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -B 
#BSUB -N
#BSUB -u s250062@dtu.dk
#BSUB -R "span[hosts=1]"
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
python train.py --note "added droput 0.3 to every layer"