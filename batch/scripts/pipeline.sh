### Example of batch script

#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc

### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm
#BSUB -W 10:00
### request 20GB of system-memory
#BSUB -R "rusage[mem=20GB]"

### -- set the job Name --
#BSUB -J lasernet
### -- Specify the output and error file. %J is the job-id --
#BSUB -o batch/logs/lasernet%J.out
#BSUB -e batch/logs/lasernet%J.err

# -- end of LSF options --

uv sync

make hpc