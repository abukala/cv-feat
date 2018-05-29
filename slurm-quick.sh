#!/bin/bash

#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --output=quick.out

module add plgrid/tools/python/3.5.2

python3 -m memory_profiler ${1} ${@:2}