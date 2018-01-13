import multiprocessing as mp

from databases import pull_pending, submit_result
from datasets import load
from features import get_sift, get_surf, get_hog, normalize_hist, get_pix

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import logging
logger = logging.getLogger('runner')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-4s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

N_PROCESSES = 24


def run():
    while True:
        trial = pull_pending()

        if trial is None:
            break

        logger.info("Starting - dataset: %s - feature: %s - clf: %s" % (trial['Dataset'],
                                                                         trial['Feature'],
                                                                         trial['Classifier']))

        assert trial['Classifier'] in ['KNN', 'RFC', 'SVM', 'LDA']
        assert trial['Feature'] in ['sift', 'surf', 'hog', 'none']

        params = eval(trial['Parameters'])

        clf = eval(trial['Classifier'])(params['clf_params'])

        (X_train, y_train), (X_test, y_test) = load(trial['Dataset'])

        feature = trial['Feature']
        if feature == 'sift':
            X_train, kmeans = get_sift(X_train)
            X_test, _ = get_sift(X_test, kmeans)
            X_train, X_test = normalize_hist(X_train, X_test)
        elif feature == 'surf':
            X_train, kmeans = get_surf(X_train)
            X_test, _ = get_surf(X_test, kmeans)
            X_train, X_test = normalize_hist(X_train, X_test)
        elif feature == 'hog':
            X_train = get_hog(X_train)
            X_test = get_hog(X_test)
        elif feature == 'none':
            X_train = get_pix(X_train)
            X_test = get_pix(X_test)

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, predictions)

        logger.info("Finished - dataset: %s - feature: %s - clf: %s - score: %s" % (trial['Dataset'],
                                                                                     trial['Feature'],
                                                                                     trial['Classifier'],
                                                                                     score))

        submit_result(trial, score)


if __name__ == '__main__':
    for _ in range(N_PROCESSES):
        mp.Process(target=run).start()
