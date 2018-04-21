import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import rescale
import logging
logger = logging.getLogger(__name__)


def get_hog(X, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    if len(X.shape) > 3:
        X = [rgb2gray(img) for img in X]
    X = [hog(img, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm="L1-sqrt") for img in X]

    return X




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


def get_pix(X, scale=False, max_value=1.0):
    # Rescale images
    if scale:
        X = [rescale(img, scale) for img in X]
    # Normalize images
    if max_value == 255:
        X = [img/255 for img in X]
    # Flatten data
    X = [x.flatten() for x in X]

    return X


def normalize_hist(X_train, X_test):
    max_value = max(np.concatenate(X_train))
    X_train = [hist/max_value for hist in X_train]
    X_test = [hist/max_value for hist in X_test]

    return X_train, X_test

# def get_surf(X, kmeans=None, k=500, multi=100, n_init=1, max_iter=10):
#     surf = cv2.xfeatures2d.SURF_create()
#     descriptors = []
#     for image in X:
#         _, des = surf.detectAndCompute(image, None)
#         if des:
#             descriptors.append(des)
#         else:
#             descriptors.append([])
#
#     hist, kmeans = _get_hist(descriptors, kmeans, k, multi, n_init, max_iter)
#
#     return hist, kmeans


# def get_sift(X, kmeans=None, k=500, multi=100, n_init=1, max_iter=10):
#     sift = cv2.xfeatures2d.SIFT_create()
#     descriptors = []
#     for image in X:
#         _, des = sift.detectAndCompute(image, None)
#         if des:
#             descriptors.append(des)
#         else:
#             descriptors.append([])
#
#     hist, kmeans = _get_hist(descriptors, kmeans, k, multi, n_init, max_iter)
#
#     return hist, kmeans
