import multiprocessing as mp

from databases import pull_pending, submit_result
from datasets import cifar10, stl10, gtsrb, mnist, feret
from features import get_hog, get_pix
from noise import apply_noise
from denoise import denoise

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

N_PROCESSES = 5

noise_params = {
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


def get_noise_params(noise_type):
    n_params = noise_params[noise_type]
    return np.arange(n_params['min']+n_params['step'], n_params['max'] + n_params['step'], n_params['step'])


def run():
    while True:
        trial = pull_pending()

        if trial is None:
            break

        logger.info("Starting - dataset: %s - feature: %s - clf: %s" % (trial['Dataset'],
                                                                         trial['Feature'],
                                                                         trial['Classifier']))

        assert trial['Dataset'] in ['gtsrb', 'cifar10', 'stl10', 'mnist', 'feret']
        assert trial['Classifier'] in ['KNN', 'RFC', 'SVM', 'LDA']
        assert trial['Feature'] in ['sift', 'surf', 'hog', 'none']
        assert trial['Noise_Type'] in noise_params.keys() or 'none' or 'random'
        assert trial['Train_Noise'] in ['yes', 'no']

        scale = False

        if trial['Dataset'].startswith('feret'):
            (X_train_clean, y_train), (X_test_clean, y_test) = feret.load_data()
            scale = 0.25
        else:
            ds = eval(trial['Dataset'])
            (X_train_clean, y_train), (X_test_clean, y_test) = ds.load_training_data(), ds.load_test_data()

        noise_type, noise_level, train_noise = trial['Noise_Type'], trial['Noise_Level'], trial['Train_Noise']
        params = eval(trial['Parameters'])
        feature_params = {}
        denoise_params = None
        if 'feature_params' in params:
            feature_params = params['feature_params']
        if 'denoise_params' in params:
            denoise_params = params['denoise_params']

        if noise_type != 'none' and noise_level != 'none':
            if noise_type == 'random':
                noise_types = [np.random.choice(['sp', 'gauss', 'quantization']) for _ in X_test_clean]
                noise_levels = [np.random.choice(get_noise_params(n_type)) for n_type in noise_types]
            else:
                noise_types = [noise_type for _ in X_test_clean]
                if noise_level == 'random':
                    noise_range = get_noise_params(noise_type)
                    noise_levels = [np.random.choice(noise_range) for _ in X_test_clean]
                else:
                    noise_levels = [noise_level for _ in X_test_clean]

            X_test = []
            for img, noise_type, noise_level in zip(X_test_clean, noise_types, noise_levels):
                noisy = apply_noise(img, noise_type, noise_level)
                if denoise_params:
                    denoised = denoise(noisy, denoise_params[0], denoise_params[1])
                    if denoised.max() == np.nan or denoised.min() == np.nan:
                        i = 0
                        while denoised.max() == np.nan or denoised.min == np.nan:
                            if i >= 1000:
                                logger.error('Failed to denoise image with method: %s, %s.' % (denoise_params[0], denoise_params[1]))
                                raise ValueError
                            denoised = denoise(noisy, denoise_params[0], denoise_params[1])
                            i += 1
                    X_test.append(denoised)
                X_test.append(noisy)
            X_test = np.array(X_test)

            if train_noise == 'yes':
                if noise_type == 'random':
                    noise_types = [np.random.choice(['sp', 'gauss', 'quantization']) for _ in X_train_clean]
                    noise_levels = [np.random.choice(get_noise_params(n_type)) for n_type in noise_types]
                    X_train = np.array([apply_noise(img, noise_type, noise_level) for img, noise_type, noise_level in
                                       zip(X_train_clean, noise_types, noise_levels)])
                else:
                    if noise_level == 'random':
                        noise_range = get_noise_params(noise_type)
                        noise_levels = [np.random.choice(noise_range) for _ in X_train_clean]
                        X_train = np.array(
                            [apply_noise(img, noise_type, noise_level) for img, noise_level in zip(X_train_clean, noise_levels)])
                    else:
                        X_train = np.array([apply_noise(img, noise_type, noise_level) for img in X_train_clean])
            else:
                X_train = X_train_clean
        else:
            X_train, X_test = X_train_clean, X_test_clean

        feature = trial['Feature']

        if feature == 'hog':
            X_train = get_hog(X_train, **feature_params)
            X_test = get_hog(X_test, **feature_params)
        elif feature == 'none':
            X_train = get_pix(X_train, scale=scale)
            X_test = get_pix(X_test, scale=scale)

        assert len(X_train) == len(y_train), (len(X_train), len(y_train))
        assert len(X_test) == len(y_test), (len(X_test), len(y_test))

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
