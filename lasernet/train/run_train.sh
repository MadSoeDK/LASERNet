#!/bin/bash
#BSUB -J lasernet_train
#BSUB -W 01:10
#BSUB -q gpuv100
#BSUB -o output_%J.txt
#BSUB -e error_%J.txt
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

cd ~/LASERNet
uv run lasernet/train/train.py
