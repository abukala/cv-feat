#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=24:00:00

module add plgrid/tools/python/3.5.2
module add plgrid/libs/libpng
module add plgrid/libs/libjpeg-turbo
module add plgrid/libs/fftw/3.3.7

python3 ${1} ${@:2}