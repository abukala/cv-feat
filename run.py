import multiprocessing as mp

from databases import pull_pending, submit_result
from datasets import cifar10, stl10, gtsrb, mnist
from features import get_sift, get_surf, get_hog, normalize_hist, get_pix
from noise import apply_gaussian_noise, apply_salt_and_pepper_noise, apply_quantization_noise, lower_resolution, apply_occlusion

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np

import logging
logger = logging.getLogger('runner')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-4s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

N_PROCESSES = 24

noise = {
    'gauss': apply_gaussian_noise,
    'quantization': apply_quantization_noise,
    'sp': apply_salt_and_pepper_noise,
    'lres': lower_resolution,
    'occlusion': apply_occlusion
}


def run():
    while True:
        trial = pull_pending()

        if trial is None:
            break

        logger.info("Starting - dataset: %s - feature: %s - clf: %s" % (trial['Dataset'],
                                                                         trial['Feature'],
                                                                         trial['Classifier']))

        assert trial['Dataset'] in ['gtsrb', 'cifar10', 'stl10', 'mnist']
        assert trial['Classifier'] in ['KNN', 'RFC', 'SVM', 'LDA']
        assert trial['Feature'] in ['sift', 'surf', 'hog', 'none']
        assert trial['Noise_Type'] in noise.keys() or 'none'
        assert trial['Train_Noise'] in ['yes', 'no']

        ds = eval(trial['Dataset'])
        (X_train, y_train), (X_test, y_test) = ds.load_training_data(), ds.load_test_data()

        noise_type, noise_level, train_noise = trial['Noise_Type'], trial['Noise_Level'], trial['Train_Noise']

        pre_size = X_test[0].shape
        pre_dtype = X_test[0].dtype

        assert isinstance(noise_level, str)
        noise_level = eval(noise_level)
        if noise_type == 'lres':
            noise_level = int(noise_level)
        X_test = np.array([noise[noise_type](img, noise_level) for img in X_test])

        if train_noise == 'yes':
            X_train = np.array([noise[noise_type](img, noise_level) for img in X_train])

        assert X_test[0].shape == pre_size
        assert X_test[0].dtype == pre_dtype

        feature = trial['Feature']
        params = eval(trial['Parameters'])
        feature_params = {}
        if 'feature_params' in params:
            if feature in params['feature_params']:
                feature_params = params['feature_params'][feature]

        if feature == 'sift':
            X_train, kmeans = get_sift(X_train, **feature_params)
            X_test, _ = get_sift(X_test, kmeans, **feature_params)
            X_train, X_test = normalize_hist(X_train, X_test)
        elif feature == 'surf':
            X_train, kmeans = get_surf(X_train, **feature_params)
            X_test, _ = get_surf(X_test, kmeans, **feature_params)
            X_train, X_test = normalize_hist(X_train, X_test)
        elif feature == 'hog':
            X_train = get_hog(X_train, **feature_params)
            X_test = get_hog(X_test, **feature_params)
        elif feature == 'none':
            X_train = get_pix(X_train)
            X_test = get_pix(X_test)

        clf_params = params['clf_params']
        clf = eval(trial['Classifier'])(**clf_params)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, predictions)

        logger.info("Finished - dataset: %s - feature: %s - clf: %s noise: (%s, %s) - score: %s" % (trial['Dataset'], trial['Feature'], trial['Classifier'], trial['Noise_Type'], trial['Noise_Level'], score))

        submit_result(trial, score)


if __name__ == '__main__':
    for _ in range(N_PROCESSES):
        mp.Process(target=run).start()
