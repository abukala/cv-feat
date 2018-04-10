from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from datasets import stl10, gtsrb, mnist, feret
from features import get_hog
import sys
import json
import time
import pathlib

RESULTS_DIR = pathlib.Path() / 'results'

hog_params = {
    'stl10': {
        'pixels_per_cell': [(8, 8), (10, 10), (12, 12), (16, 16)]
    },
    'gtsrb': {
        'pixels_per_cell': [(3, 3), (4, 4), (6, 6), (8, 8)]
    },
    'mnist': {
        'pixels_per_cell': [(3, 3), (4, 4), (6, 6)]
    },
    'feret': {
        'pixels_per_cell': [(24, 24), (28, 28), (32, 32), (40, 40), (48, 48)]
    }
}

clf_params = {
    'SVM': {
        'C': [0.001, 0.1, 1, 10, 100]
    },
    'RFC': {
        'n_estimators': [5, 10, 25, 50, 100],
    },
    'LDA': {
        'n_components': [1, 2, 4, 6, 8]
    },
    'KNN': {
        'n_neighbors': [1, 5, 10, 25, 50]
    }
}

if __name__ == '__main__':
    assert len(sys.argv) == 4
    assert sys.argv[1] in ['stl10', 'gtsrb', 'mnist', 'feret']
    dataset = eval(sys.argv[1])
    cells_per_block = (int(sys.argv[2]), int(sys.argv[2]))
    clf_label = sys.argv[3]
    X, y = dataset.load_training_data()
    results = []

    for pixels_per_cell in reversed(hog_params[sys.argv[1]]['pixels_per_cell']):
        ts = time.time()
        des = get_hog(X, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        estimator = eval(clf_label)()
        clf = GridSearchCV(estimator=estimator, param_grid=clf_params[clf_label], cv=5, n_jobs=5)
        clf.fit(des, y)
        job_time = round((time.time() - ts) / 60, 1)
        results.append({
            "pixels_per_cell": pixels_per_cell,
            "cells_per_block": cells_per_block,
            "clf": clf_label,
            "clf_params": clf.best_params_,
            "score": clf.best_score_,
            "job_time": job_time
        })

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir()
    filename = '%s.json' % sys.argv[1]
    filepath = RESULTS_DIR / filename
    with filepath.open(mode='a') as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write("\n")