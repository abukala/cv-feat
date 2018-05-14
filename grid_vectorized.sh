#!/bin/bash

for dataset in "stl10" "gtsrb" "feret" "mnist"
do
    for cls in "KNN" "SVM" "RFC" "LDA"
    do
        sbatch slurm-grid.sh grid_clf.py $dataset $cls
    done
done
