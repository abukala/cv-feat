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

for dataset in ['gtsrb']:
    for feature in ['hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            for orientations in [8, 9]:
                for nums in [(1, 2), (1, 4), (1, 6), (2, 2), (2, 4), (2, 6), (4, 2), (4, 4), (4, 1), (2, 1)]:
                    trial = {
                        'Dataset': dataset,
                        'Feature': feature,
                        'Parameters': {
                            'clf_params': parameters[classifier],
                            'feature_params': {
                                'hog': {
                                    'pixels_per_cell': (nums[0], nums[0]),
                                    'cells_per_block': (nums[1], nums[1]),
                                    'orientations': orientations
                                }
                            }
                        },
                        'Classifier': classifier
                    }
                    databases.add_to_pending(trial)
