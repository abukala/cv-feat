#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00

module add plgrid/tools/python/3.6.0

python3 ${1}