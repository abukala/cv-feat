#!/bin/bash

#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --output=quick.out

module add plgrid/tools/python/3.6.0

python3 ${1} ${@:2}