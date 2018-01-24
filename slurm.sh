#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00

env/bin/python ${1}