#!/bin/bash
#BSUB -J micro-net-13
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
##BSUB -u s211548@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/micro-net_13_%J.out
#BSUB -e logs/micro-net_13_%J.err

# Load modules
module load python3/3.12.0
module load cuda/12.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Run the microstructure training script with early stopping
python train_micro_net_cnn_lstm.py \
  --epochs 200 \
  --batch-size 16 \
  --lr 1e-3 \
  --seq-length 4 \
  --plane xz \
  --split-ratio "12,6,6" \
  --use-weighted-loss \
  --loss-type combined \
  --T-solidus 1000 \
  --T-liquidus 2500
