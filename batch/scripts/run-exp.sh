### Example of batch script

#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"

### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm
#BSUB -W 10:00
### Email address to notify
### request system-memory
#BSUB -R "rusage[mem=24GB]"
### email when job begins (optional)
#BSUB -B
### email when job ends             
#BSUB -N
#BSUB -u s215805@dtu.dk

### -- set the job Name --
#BSUB -J lasernet
### -- Specify the output and error file. %J is the job-id --
#BSUB -o batch/logs/lasernet%J.out
#BSUB -e batch/logs/lasernet%J.err

# -- end of LSF options --

uv sync

export WANDB_PROJECT="lasernet" 

make run-exp