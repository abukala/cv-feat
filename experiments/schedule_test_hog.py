import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases

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

for dataset in ['cifar10', 'stl10', 'mnist']:
    for feature in ['hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            for orientations in [8, 9, 10, 11]:
                for ppc in range(1, 7):
                    for cpb in range(1, 7):
                        if dataset == 'cifar10' and ppc*cpb > 32:
                            continue
                        if dataset == 'mnist' and ppc*cpb > 28:
                            continue
                        trial = {
                            'Dataset': dataset,
                            'Feature': feature,
                            'Parameters': {
                                'clf_params': parameters[classifier],
                                'feature_params': {
                                    'hog': {
                                        'pixels_per_cell': (ppc, ppc),
                                        'cells_per_block': (cpb, cpb),
                                        'orientations': orientations
                                    }
                                }
                            },
                            'Classifier': classifier
                        }
                        databases.add_to_pending(trial)
