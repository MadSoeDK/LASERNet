#!/bin/bash
#BSUB -J MICROnet_notebook
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
##BSUB -u s211548@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/MICROnet_notebook_%J.out
#BSUB -e logs/MICROnet_notebook_%J.err

# Load modules
module load python3/3.12.0
module load cuda/12.4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Run the MICROnet_notebook
make MICROnet_notebook