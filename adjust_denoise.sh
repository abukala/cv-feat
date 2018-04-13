#!/bin/bash

for dataset in "stl10" "gtsrb" "feret" "mnist"
do
    for noise in "gauss" "sp" "quantization" "blur" "random"
    do
        sbatch slurm.sh denoise.py $dataset $noise
    done
done