#!/bin/bash
#BSUB -J name
#BSUB -q hpc
#BSUB -W 2
#BSUB -R ”rusage[mem=512MB]”
#BSUB –n 4
#BSUB –R ”span[hosts=1]”
#BSUB -o name_%J.out
#BSUB -e name_%J.err

