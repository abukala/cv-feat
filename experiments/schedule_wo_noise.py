from .. import databases
from .. import datasets

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

for dataset in datasets.DATASET_NAMES:
    for feature in ['none', 'sift', 'surf', 'hog']:
        for classifier in ['KNN', 'LDA', 'SVM', 'RFC']:
            trial = {
                'Dataset': dataset,
                'Feature': feature,
                'Parameters': {
                    'clf_params': parameters[classifier]
                },
                'Classifier': classifier
            }
            databases.add_to_pending(trial)
