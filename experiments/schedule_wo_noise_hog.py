import sys
import os

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

for dataset in ['gtsrb', 'stl10', 'cifar10', 'mnist']:
    for feature in ['none', 'hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            for noise_type in ['gauss', 'sp', 'quantization', 'lres', 'occlusion']:
                for train_noise in ['yes', 'no']:
                    noise_level = 0
                    if noise_type == 'lres':
                        noise_level = 1
                    trial = {
                        'Dataset': dataset,
                        'Feature': feature,
                        'Parameters': {
                            'clf_params': parameters[classifier],
                            'feature_params': feature_params[dataset]
                        },
                        'Noise_Type': noise_type,
                        'Noise_Level': noise_level,
                        'Train_Noise': train_noise,
                        'Classifier': classifier
                    }
                    databases.add_to_pending(trial)
