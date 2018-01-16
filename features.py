import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.feature import hog
import logging
logger = logging.getLogger(__name__)


def get_hog(X, pixels_per_cell=(2, 2), cells_per_block=(1, 1), orientations=12, block_norm='L1-sqrt'):
    if len(X.shape) > 3:
        X = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X]
    X = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                   block_norm=block_norm) for img in X]

    return X


def get_surf(X, kmeans=None, k=500, multi=100, n_init=1, max_iter=10):
    surf = cv2.xfeatures2d.SURF_create()
    descriptors = []
    for image in X:
        _, des = surf.detectAndCompute(image, None)
        if des is not None:
            descriptors.append(des)
        else:
            descriptors.append([])

    hist, kmeans = _get_hist(descriptors, kmeans, k, multi, n_init, max_iter)

    return hist, kmeans


def _get_hist(X, kmeans=None, k=500, multi=100, n_init=1, max_iter=10):
    X_list = []
    for group in X:
        X_list.extend(group)
    X_list = np.array(X_list)

    if k*multi < len(X_list):
        choice = X_list[np.random.choice(len(X_list), k*multi, replace=False)]
    else:
        choice = X_list

    if not kmeans:
        kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter).fit(choice)

    predictions = kmeans.predict(X_list)
    idx = []
    for group in X:
        des_idx = []
        for _ in group:
            des_idx.append(predictions[0])
            predictions = np.delete(predictions, 0)
        idx.append(des_idx)

    bins = np.arange(k)
    hist = [np.histogram(i, bins=bins)[0] for i in idx]

    return hist, kmeans


def get_sift(X, kmeans=None, k=500, multi=100, n_init=1, max_iter=10):
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors = []
    for image in X:
        _, des = sift.detectAndCompute(image, None)
        if des is not None:
            descriptors.append(des)
        else:
            descriptors.append([])

    hist, kmeans = _get_hist(descriptors, kmeans, k, multi, n_init, max_iter)

    return hist, kmeans


def get_pix(X):
    # Normalize images
    X = _normalize_images(X)
    # Flatten data
    X = [x.flatten() for x in X]

    return X


def _normalize_images(X):
    v_min = X.min(axis=(0, 1), keepdims=True)
    v_max = X.max(axis=(0, 1), keepdims=True)

    return (X - v_min)/(v_max - v_min)


def normalize_hist(X_train, X_test):
    max_value = max(np.concatenate(X_train))
    X_train = [hist/max_value for hist in X_train]
    X_test = [hist/max_value for hist in X_test]

    return X_train, X_test

