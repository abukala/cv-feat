import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import databases
from utils import get_params

params = get_params()

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
    }
}

for dataset in ['gtsrb', 'stl10', 'feret', 'mnist']:
    for feature in ['none', 'hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            for noise_type in noise.keys():
                for train_noise in ['no', 'yes']:
                    nr = noise[noise_type]
                    noise_level_range = np.arange(nr['min']+nr['step'], nr['max']+nr['step'], nr['step'])
                    for num in noise_level_range:
                        num = round(num, 3)
                        if num not in params[dataset]['denoise'][noise_type]:
                            print("Missing noise level in params: %s, %s, %s" % (dataset, noise_type, num))
                            continue
                        trial = {
                            'Dataset': dataset,
                            'Feature': feature,
                            'Parameters': {
                                'clf_params': params[dataset]['clf'][feature][classifier],
                                'feature_params': params[dataset]['hog'],
                                'denoise_params': params[dataset]['denoise'][noise_type][num]
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
                        'Feature': feature,
                        'Parameters': {
                            'clf_params': params[dataset]['clf'][feature][classifier],
                            'feature_params': params[dataset]['hog'],
                            'denoise_params': params[dataset]['denoise'][noise_type]['random']
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
                        'Feature': feature,
                        'Parameters': {
                            'clf_params': params[dataset]['clf'][feature][classifier],
                            'feature_params': params[dataset]['hog'],
                            'denoise_params': params[dataset]['denoise']['random']
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
                'Feature': feature,
                'Parameters': {
                    'clf_params': params[dataset]['clf'][feature][classifier],
                    'feature_params': params[dataset]['hog']
                },
                'Noise_Type': 'none',
                'Noise_Level': 'none',
                'Train_Noise': 'no',
                'Classifier': classifier
            }
            databases.add_to_pending(trial)