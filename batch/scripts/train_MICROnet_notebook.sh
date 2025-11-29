#!/bin/bash
#BSUB -J micro-net-1
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
##BSUB -u s211548@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/micro-net_1_%J.out
#BSUB -e logs/micro-net_1_%J.err

# Load modules
module load python3/3.12.0
module load cuda/12.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Run the MICROnet_notebook
make MICROnet_notebook