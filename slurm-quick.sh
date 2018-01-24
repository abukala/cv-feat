#!/bin/bash

#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --output=quick.out

env/bin/python ${1} ${@:2}