#!/bin/bash

for dataset in "stl10" "gtsrb" "feret" "mnist"
do
    for cells_per_block in 1 2 3
    do
        for cls in "KNN" "SVM" "RFC" "LDA"
        do
            sbatch slurm-adjust.sh $dataset $cells_per_block $cls
        done
    done
done
