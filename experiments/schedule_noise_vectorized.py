import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases

parameters = {
    'gtsrb': {
        'clf': {
            'RFC': {
                'n_estimators': 100
            },
            'SVM': {
                'C': 100
            },
            'LDA': {
                'n_components': 1
            },
            'KNN': {
                'n_neighbors': 1
            }
        }
    },
    'stl10': {
        'clf': {
            'SVM': {
                'C': 10
            },
            'RFC': {
                'n_estimators': 100
            },
            'LDA': {
                'n_components': 1
            },
            'KNN': {
                'n_neighbors': 1
            }
        }
    },
    'mnist': {
        'clf': {
            'RFC': {
                'n_estimators': 100
            },
            'SVM': {
                'C': 100
            },
            'LDA': {
                'n_components': 1
            },
            'KNN': {
                'n_neighbors': 1
            }
        }
    },
    'feret': {
        'clf': {
            'RFC': {
                'n_estimators': 100
            },
            'SVM': {
                'C': 100
            },
            'LDA': {
                'n_components': 1
            },
            'KNN': {
                'n_neighbors': 1
            }
        }
    },
}

noise = {
    'gauss': {
        'min': 0,
        'max': 0.25,
        'step': 0.025
    },
    'quantization': {
        'min': 0,
        'max': 0.5,
        'step': 0.05
    },
    'sp': {
        'min': 0,
        'max': 0.2,
        'step': 0.02
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
    for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
        for noise_type in noise.keys():
            for train_noise in ['no', 'yes']:
                nr = noise[noise_type]
                noise_level_range = np.arange(nr['min']+nr['step'], nr['max']+nr['step'], nr['step'])
                for num in noise_level_range:
                    trial = {
                        'Dataset': dataset,
                        'Feature': 'none',
                        'Parameters': {
                            'clf_params': parameters[dataset]['clf'][classifier],
                        },
                        'Noise_Type': noise_type,
                        'Noise_Level': num,
                        'Train_Noise': train_noise,
                        'Classifier': classifier
                    }
                    databases.add_to_pending(trial)
                # Known noise with random intensity
                trial = {
                    'Dataset': dataset,
                    'Feature': 'none',
                    'Parameters': {
                        'clf_params': parameters[dataset]['clf'][classifier],
                    },
                    'Noise_Type': noise_type,
                    'Noise_Level': 'random',
                    'Train_Noise': train_noise,
                    'Classifier': classifier
                }
                databases.add_to_pending(trial)
                # Unknown noise with unknown intensity
                trial = {
                    'Dataset': dataset,
                    'Feature': 'none',
                    'Parameters': {
                        'clf_params': parameters[dataset]['clf'][classifier],
                    },
                    'Noise_Type': 'random',
                    'Noise_Level': 'random',
                    'Train_Noise': train_noise,
                    'Classifier': classifier
                }
                databases.add_to_pending(trial)
        # Clean conditions
        trial = {
            'Dataset': dataset,
            'Feature': 'none',
            'Parameters': {
                'clf_params': parameters[dataset]['clf'][classifier],
                'feature_params': parameters[dataset]['hog']
            },
            'Noise_Type': 'none',
            'Noise_Level': 'none',
            'Train_Noise': 'no',
            'Classifier': classifier
        }
        databases.add_to_pending(trial)