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
            'pixels_per_cell': (4, 4),
            'cells_per_block': (4, 4),
        }
    },
    'mnist': {
        'hog': {
            'pixels_per_cell': (4, 4),
            'cells_per_block': (3, 3),
        }
    },
    'stl10': {
        'hog': {
            'pixels_per_cell': (6, 6),
            'cells_per_block': (2, 2),
        }
    },
    'feret': {
        'hog': {
            'pixels_per_cell': (24, 24),
            'cells_per_block': (1, 1)
        }
    }
}


for dataset in ['gtsrb', 'stl10', 'feret', 'mnist']:
    for feature in ['none', 'hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            for train_noise in ['no', 'yes']:
                nr = {
                    'min': 0,
                    'max': 5,
                    'step': 0.5
                }
                for num in np.arange(nr['min'], nr['max']+nr['step'], nr['step']):
                    trial = {
                        'Dataset': dataset,
                        'Feature': feature,
                        'Parameters': {
                            'clf_params': parameters[classifier],
                            'feature_params': feature_params[dataset]
                        },
                        'Noise_Type': 'blur',
                        'Noise_Level': num,
                        'Train_Noise': train_noise,
                        'Classifier': classifier,
                        'Description': 'blur_fix'
                    }
                    databases.add_to_pending(trial)