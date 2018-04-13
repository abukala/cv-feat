#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH --time=72:00:00

module add plgrid/libs/fftw/3.3.7
module add plgrid/tools/python/3.5.2

python3 ${1} ${@:2}