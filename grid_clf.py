from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from datasets import stl10, gtsrb, mnist, feret
from features import get_hog, get_pix
import sys
import json
import pathlib

RESULTS_DIR = pathlib.Path() / 'results' / 'baseline'

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
    assert len(sys.argv) == 3
    assert sys.argv[1] in ['stl10', 'gtsrb', 'mnist', 'feret']
    assert sys.argv[2] in ['KNN', 'SVM', 'LDA', 'RFC']
    dataset = eval(sys.argv[1])
    estimator = eval(sys.argv[2])()
    X, y = dataset.load_training_data()
    clf = GridSearchCV(estimator=estimator, param_grid=clf_params[sys.argv[2]], cv=5, n_jobs=5)
    scale = None
    if sys.argv[1] == 'feret':
        scale = 0.25
    clf.fit(get_pix(X, scale=scale), y)
    results = {
        "clf": sys.argv[2],
        "clf_params": clf.best_params_,
        "score": clf.best_score_
    }

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir()
    filename = '%s_vectorized.json' % sys.argv[1]
    filepath = RESULTS_DIR / filename
    with filepath.open(mode='a') as outfile:
        json.dump(results, outfile)
        outfile.write("\n")