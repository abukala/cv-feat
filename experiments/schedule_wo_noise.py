import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases
import datasets

parameters = {
    'KNN': {
        'n_neighbors': 10
    },
    'LDA': {
        'tol': 0.1
    },
    'SVM': {
        'kernel': 'linear',
        'C': 0.1,
        'tol': 0.01
    },
    'RFC': {
        'n_estimators': 100
    }
}

feature_params = {
    'gtsrb': {
        'hog': {
            'pixels_per_cell': (5, 5),
            'cells_per_block': (6, 6),
            'orientations': 8
        }
    },
    'mnist': {},
    'stl10': {},
    'cifar10': {}
}

for dataset in datasets.DATASET_NAMES:
    for feature in ['none', 'sift', 'surf', 'hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            trial = {
                'Dataset': dataset,
                'Feature': feature,
                'Parameters': {
                    'clf_params': parameters[classifier],
                    'feature_params': feature_params[dataset]
                },
                'Classifier': classifier
            }
            databases.add_to_pending(trial)
