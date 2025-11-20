#!/bin/bash
#BSUB -J lasernet-microstructure
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
##BSUB -u s215805@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/lasernet_micro_%J.out
#BSUB -e logs/lasernet_micro_%J.err

# Load modules
module load python3/3.12.0
module load cuda/12.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Run the microstructure training script
python train_microstructure.py \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --seq-length 3 \
  --plane xz \
  --split-ratio "12,6,6"
