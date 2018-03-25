import sys
import os
import numpy as np
from ..datasets.feret import params as feret_ft_params

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
    'feret': feret_ft_params
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
    'blur': {
        'min': 0,
        'max': 5,
        'step': 0.5
    },
    'occlusion': {
        'min': 0,
        'max': 0.8,
        'step': 0.1
    }
}

for dataset in ['gtsrb', 'stl10', 'feret', 'mnist']:
    for feature in ['none', 'hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            for noise_type in noise:
                for train_noise in ['no', 'yes']:
                    nr = noise[noise_type]
                    for num in np.arange(nr['min'], nr['max']+nr['step'], nr['step']):
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
                            'Classifier': classifier,
                            'Description': 'full_noise_hog'
                        }
                        databases.add_to_pending(trial)