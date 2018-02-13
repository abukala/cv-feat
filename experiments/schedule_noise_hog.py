import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases

parameters = {
    'KNN': {
        'n_neighbors': 5
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
            'pixels_per_cell': (2, 2),
            'cells_per_block': (4, 4),
            'orientations': 9
        }
    },
    'mnist': {
        'hog': {
            'pixels_per_cell': (4, 4),
            'cells_per_block': (3, 3),
            'orientations': 10
        }
    },
    'stl10': {
        'hog': {
            'pixels_per_cell': (6, 6),
            'cells_per_block': (3, 3),
            'orientations': 11
        }
    },
    'cifar10': {
        'hog': {
            'pixels_per_cell': (4, 4),
            'cells_per_block': (3, 3),
            'orientations': 10
        }
    }
}

noise = {
    'gauss': {
        'min': 0,
        'max': 0.5,
        'step': 0.05
    },
    'quantization': {
        'min': 0,
        'max': 1,
        'step': 0.1
    },
    'sp': {
        'min': 0,
        'max': 0.25,
        'step': 0.025
    },
    'lres': {
        'min': 1,
        'max': 9,
        'step': 1
    },
    'occlusion': {
        'min': 0,
        'max': 0.8,
        'step': 0.1
    }
}

for dataset in ['gtsrb', 'stl10', 'cifar10', 'mnist']:
    for feature in ['none', 'hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            for noise_type in noise:
                for train_noise in ['no', 'yes']:
                    nr = noise[noise_type]
                    for num in np.arange(nr['min'], nr['max'], nr['step']):
                        trial = {
                            'Dataset': dataset,
                            'Feature': feature,
                            'Parameters': {
                                'clf_params': parameters[classifier],
                                'feature_params': feature_params[dataset]
                            },
                            'Noise_Type': noise_type,
                            'Noise_Level': num,
                            'Train_Noise': train_noise,
                            'Classifier': classifier
                        }
                        databases.add_to_pending(trial)