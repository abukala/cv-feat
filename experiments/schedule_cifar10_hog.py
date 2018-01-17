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

for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
    for pixels_per_cell in [(num, num) for num in range(1, 10)]:
        for cells_per_block in [(num, num) for num in range(1, 20)]:
            for orientations in [8, 9, 10, 11, 12]:
                if pixels_per_cell[0] * cells_per_block[0] > 32:
                    continue
                trial = {
                    'Dataset': 'cifar10',
                    'Feature': 'hog',
                    'Parameters': {
                        'clf_params': parameters[classifier],
                        'feature_params': {
                            'pixels_per_cell': pixels_per_cell,
                            'cells_per_block': cells_per_block,
                            'orientations': orientations
                        }
                    },
                    'Classifier': classifier
                }
                databases.add_to_pending(trial)
