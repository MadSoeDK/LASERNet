#!/bin/bash
#BSUB -J micro-net-predrnn-8h-4c
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
##BSUB -u s211548@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/micro-predrnn_%J.out
#BSUB -e logs/micro-predrnn_%J.err

# Load modules
module load python3/3.12.0
module load cuda/12.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Run the microstructure PredRNN training script with early stopping
python train_micro_net_predrnn.py \
  --epochs 200 \
  --batch-size 16 \
  --lr 1e-3 \
  --seq-length 3 \
  --plane xz \
  --rnn-layers 4 \
  --split-ratio "12,6,6" \
  --use-weighted-loss \
  --T-solidus 1400.0 \
  --T-liquidus 1500.0 \
  --weight-scale 0.1 \
  --base-weight 0.1
