#!/bin/bash
#BSUB -J lasernet
#BSUB -q c02516
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:45
#BSUB -R "rusage[mem=3GB]"
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
python train.py --epoch 300 --note "BASELINE + U-Net + convo 3-->4, epoch300, lstm layer, seq2 + lr 2e-4, added clipped gradient+sched, shuffle true" --seq-length 2 --lr 2e-4