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

if __name__ == '__main__':
    for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
        for ratio in [60, 70, 80]:
            for ppc, cpb in [(28, 1), (28, 2), (24, 1), (24, 2), (20, 1), (16, 1)]:
                trial = {
                    'Dataset': 'feret%s' % ratio,
                    'Feature': 'hog',
                    'Parameters': {
                        'clf_params': parameters[classifier],
                        'feature_params': {
                            'hog': {
                                'pixels_per_cell': (ppc, ppc),
                                'cells_per_block': (cpb, cpb)
                            }
                        }
                    },
                    'Noise_Type': 'none',
                    'Noise_Level': 'none',
                    'Train_Noise': 'no',
                    'Classifier': classifier,
                    'Description': 'test_feret%s' % ratio
                }
                databases.add_to_pending(trial)