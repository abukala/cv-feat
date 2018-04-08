#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH --time=72:00:00

module add plgrid/tools/python/3.5.2

python3 ${1} ${@:2}