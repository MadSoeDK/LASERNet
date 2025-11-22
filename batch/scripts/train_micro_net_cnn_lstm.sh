#!/bin/bash
#BSUB -J micro-net-cnn-lstm
#BSUB -q gpua100
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
##BSUB -u s211548@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/micro-net_%J.out
#BSUB -e logs/micro-net_%J.err

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
  --seq-length 3 \
  --plane xz \
  --split-ratio "12,6,6"
