import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.feature import hog
import logging
import sys
import os
import urllib.request
import tarfile
import zipfile
logger = logging.getLogger(__name__)


class BaseLoader:
    def __init__(self, **kwargs):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        self.size = kwargs.get('size')

    def load_data(self):
        logger.error("Load data function not overriden, exiting...")
        sys.exit()

    def get_hog(self, pixels_per_cell=(5, 5), cells_per_block=(6, 6), orientations=8):
        logger.debug("Preparing training data...")
        x_train = [cv2.cvtColor(cv2.resize(self.crop_sq(img), self.size), cv2.COLOR_BGR2GRAY) for img in self.x_train]
        x_train = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                       block_norm='L1-sqrt', feature_vector=True) for img in x_train]

        logger.debug("Descriptor length: %s" % len(x_train[0]))

        logger.debug("Preparing test data...")
        x_test = [cv2.cvtColor(cv2.resize(self.crop_sq(img), self.size), cv2.COLOR_BGR2GRAY) for img in self.x_test]
        x_test = [hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                      block_norm='L1-sqrt', feature_vector=True) for img in x_test]

        return (x_train, self.y_train), (x_test, self.y_test)

    def get_surf(self, k=500, multi=100, n_init=1, max_iter=10):
        logger.debug("Generating surf hist...")
        # Preparing training data
        des_train = self.get_surf_des(self.x_train)
        des_train_list = self.group_to_list(des_train)
        if k*multi < len(des_train_list):
            des_choice = des_train_list[np.random.choice(len(des_train_list), k*multi, replace=False)]
        else:
            des_choice = des_train_list
        kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter).fit(des_choice)
        idx_train = self.group_indices(kmeans.predict(des_train_list), des_train)
        bins = np.arange(k)
        hist_train = [np.histogram(i, bins=bins)[0] for i in idx_train]

        # Preparing test data
        des_test = self.get_surf_des(self.x_test)
        des_test_list = self.group_to_list(des_test)
        idx_test = self.group_indices(kmeans.predict(des_test_list), des_test)
        hist_test = [np.histogram(i, bins=bins)[0] for i in idx_test]

        x_train, x_test = self.normalize_hist(hist_train, hist_test)

        return (x_train, self.y_train), (x_test, self.y_test)

    def get_sift(self, k=500, multi=100, n_init=1, max_iter=10):
        logger.debug("Generating sift hist...")
        # Preparing training data
        des_train = self.get_sift_des(self.x_train)
        des_train_list = self.group_to_list(des_train)
        if k*multi < len(des_train_list):
            des_choice = des_train_list[np.random.choice(len(des_train_list), k*multi, replace=False)]
        else:
            des_choice = des_train_list
        kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter).fit(des_choice)
        idx_train = self.group_indices(kmeans.predict(des_train_list), des_train)
        bins = np.arange(k)
        hist_train = [np.histogram(i, bins=bins)[0] for i in idx_train]

        # Preparing test data
        des_test = self.get_sift_des(self.x_test)
        des_test_list = self.group_to_list(des_test)
        idx_test = self.group_indices(kmeans.predict(des_test_list), des_test)
        hist_test = [np.histogram(i, bins=bins)[0] for i in idx_test]

        x_train, x_test = self.normalize_hist(hist_train, hist_test)

        return (x_train, self.y_train), (x_test, self.y_test)

    def get_pix(self):
        # Resize images
        x_train = [cv2.resize(self.crop_sq(image), self.size) for image in self.x_train]
        x_test = [cv2.resize(self.crop_sq(image), self.size) for image in self.x_test]
        # Normalize images
        x_train = [cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for img in x_train]
        x_test = [cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for img in x_test]
        # Flatten data
        x_train = [x.flatten() for x in x_train]
        x_test = [x.flatten() for x in x_test]

        return (x_train, self.y_train), (x_test, self.y_test)

    def get_sift_des(self, images):
        sift = cv2.xfeatures2d.SIFT_create()
        descriptors = []
        for image in images:
            _, des = sift.detectAndCompute(image, None)
            if des is not None:
                descriptors.append(des)
            else:
                descriptors.append([])
        return descriptors

    def get_surf_des(self, images):
        surf = cv2.xfeatures2d.SURF_create()
        descriptors = []
        for image in images:
            _, des = surf.detectAndCompute(image, None)
            if des is not None:
                descriptors.append(des)
            else:
                descriptors.append([])
        return descriptors

    def group_indices(self, idx, des):
        group_idx = []
        for group in des:
            des_idx = []
            for des in group:
                des_idx.append(idx[0])
                idx = np.delete(idx, 0)
            group_idx.append(des_idx)

        return group_idx

    def group_to_list(self, group):
        l = []
        for g in group:
            l.extend(g)
        return np.array(l)

    def normalize_hist(self, train_data, test_data):
        max_value = max(np.concatenate(train_data))
        train_data = [hist/max_value for hist in train_data]
        test_data = [hist/max_value for hist in test_data]

        return train_data, test_data

    def crop_sq(self, image):
        height, width = image.shape[:2]
        if height < width:
            offset = int((width - height) / 2)
            image = image[:, offset:(height + offset)]
        elif height > width:
            offset = int((height - width) / 2)
            image = image[offset:(width + offset), :]
        assert image.shape[0] == image.shape[1], "Cropping didnt work, shape: %s" % image.shape[:2]
        return image

    def download_and_extract(self, url, path, dest_path=None):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = url.split('/')[-1]
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                              float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(url, filepath, reporthook=_progress)
            if not dest_path:
                dest_path = path

            if filepath.split('.')[-2] == 'tar':
                tarfile.open(filepath, 'r:gz').extractall(dest_path)
            elif filepath.split('.')[-1] == 'zip':
                zipfile.ZipFile(filepath, 'r').extractall(dest_path)

            sys.stdout.write("\n Data extracted to %s" % dest_path)