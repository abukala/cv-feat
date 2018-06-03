#!/bin/bash

for dataset in "stl10" "gtsrb" "feret" "mnist"
do
    for noise_level in 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 "random"
    do
        sbatch slurm.sh denoise.py $dataset gauss $noise_level
    done
    for noise_level in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 "random"
    do
        sbatch slurm.sh denoise.py $dataset quantization $noise_level
    done
    for noise_level in 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 "random"
    do
        sbatch slurm.sh denoise.py $dataset sp $noise_level
    done
    sbatch slurm.sh denoise.py $dataset random random
done
