#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=20GB

module add plgrid/tools/python/3.5.2

python3 ${1} ${@:2}