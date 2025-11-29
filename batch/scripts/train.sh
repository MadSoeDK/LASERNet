#!/bin/bash
#BSUB -J lasernet
#BSUB -q gpua10
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
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
python train.py --epoch 300 --note "BASELINE + U-Net + convo 3-->4, epoch300, added another lstm layer (2), peace" 