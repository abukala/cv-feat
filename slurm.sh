#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00
#SBATCH --output=slurm.out

module add plgrid/tools/python/3.5.2

python3 ${1}