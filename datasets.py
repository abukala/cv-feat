import gzip
import os
import zipfile
import numpy as np
import pickle
import cv2
import sys

import struct
import tarfile
import logging
import csv
import json

log = logging.getLogger(__name__)

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DATA_URLS = json.load(open('data_urls.json'))
DATASET_NAMES = ['gtsrb', 'cifar10', 'stl10', 'mnist']


def download(url):
    name = url.split('/')[-1]
    download_path = os.path.join(DATA_PATH, name)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\rDownloading %s %.2f%%' % (name,
                                                      float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(download_path):
        urlretrieve(url, download_path, reporthook=_progress)
        sys.stdout.write('\n')

        if name.endswith('.tar.gz'):
            tarfile.open(download_path, 'r:gz').extractall(DATA_PATH)
        elif name.endswith('.zip'):
            zipfile.ZipFile(download_path, 'r').extractall(DATA_PATH)
        elif name.endswith('.gz'):
            pass
        else:
            log.error("Unknown file type")


def load(name):
    assert name in DATASET_NAMES

    for url in DATA_URLS[name]:
        download(url)

    if name == 'gtsrb':
        train_dir = os.path.join(DATA_PATH, 'GTSRB', 'Final_Training', 'Images')
        x_train, y_train, x_test, y_test = [], [], [], []

        for (dir_path, dir_names, files) in os.walk(train_dir):
            dir_name = os.path.basename(dir_path)
            if not dir_name.isdigit():
                continue
            class_id = int(dir_name)
            for file in files:
                if file.endswith('.csv'):
                    continue
                img = cv2.imread(os.path.join(dir_path, file))
                x_train.append(img)
                y_train.append(class_id)

        test_dir = os.path.join(DATA_PATH, 'GTSRB', 'Final_Test', 'Images')
        gt_file = csv.reader(open(os.path.join(DATA_PATH, 'GT-final_test.csv')), delimiter=';')

        next(gt_file)
        for row in gt_file:
            class_id = int(row[7])
            img = cv2.imread(os.path.join(test_dir, row[0]))
            x_test.append(img)
            y_test.append(class_id)
        x_train = _resize(x_train, (40, 40))
        x_test = _resize(x_test, (40, 40))

    elif name == 'mnist':
        TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train-images-idx3-ubyte.gz')
        TRAIN_LABELS_PATH = os.path.join(DATA_PATH, 'train-labels-idx1-ubyte.gz')
        TEST_DATA_PATH = os.path.join(DATA_PATH, 't10k-images-idx3-ubyte.gz')
        TEST_LABELS_PATH = os.path.join(DATA_PATH, 't10k-labels-idx1-ubyte.gz')

        def _load_images(url):
            file = gzip.open(url, 'rb')
            _ = file.read(4)
            images_count = struct.unpack('>i', file.read(4))[0]
            rows = struct.unpack('>i', file.read(4))[0]
            cols = struct.unpack('>i', file.read(4))[0]
            l = rows * cols
            images = [np.reshape(struct.unpack('>%sB' % l, file.read(l)), (rows, cols)) for _ in range(images_count)]
            assert len(images) == images_count
            assert images[0].shape == (rows, cols)

            return images

        def _load_labels(url):
            file = gzip.open(url, 'rb')
            _ = file.read(4)
            items_count = struct.unpack('>i', file.read(4))[0]
            labels = struct.unpack('>%sB' % items_count, file.read(items_count))
            assert len(labels) == items_count

            return labels

        x_train = _load_images(TRAIN_DATA_PATH)
        x_test = _load_images(TEST_DATA_PATH)
        y_train = _load_labels(TRAIN_LABELS_PATH)
        y_test = _load_labels(TEST_LABELS_PATH)

    elif name == 'stl10':
        def _load_images(url):
            with open(url, 'rb') as f:
                everything = np.fromfile(f, dtype=np.uint8)
                images = np.reshape(everything, (-1, 3, 96, 96))
                images = np.transpose(images, (0, 3, 2, 1))
                return images

        stl10_path = os.path.join(DATA_PATH, 'stl10_binary')

        x_train = _load_images(os.path.join(stl10_path, 'train_X.bin'))
        x_test = _load_images(os.path.join(stl10_path, 'test_X.bin'))

        with open(os.path.join(stl10_path, 'train_y.bin')) as f:
            y_train = np.fromfile(f, dtype=np.uint8)
        with open(os.path.join(stl10_path, 'test_y.bin')) as f:
            y_test = np.fromfile(f, dtype=np.uint8)

    elif name == 'cifar10':
        shape = (32, 32, 3)
        train_data = []
        cifar_path = os.path.join(DATA_PATH, 'cifar-10-batches-py')
        for num in range(1, 6):
            with open(os.path.join(cifar_path, 'data_batch_%s' % num), 'rb') as f:
                train_data.append(pickle.load(f, encoding='bytes'))

        x_train = []
        y_train = []
        for batch in train_data:
            for img, label in zip(batch[b'data'], batch[b'labels']):
                x_train.append(np.reshape(img, shape))
                y_train.append(label)

        assert len(x_train) == len(y_train)
        assert x_train[0].shape == shape

        with open(os.path.join(cifar_path, 'test_batch'), 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')

        x_test = [np.reshape(img, shape) for img in test_data[b'data']]
        y_test = [label for label in test_data[b'labels']]

        assert len(x_test) == len(y_test)
        assert x_test[0].shape == shape
    else:
        raise ValueError("Unknown dataset")

    return (x_train, y_train), (x_test, y_test)


def _resize(images, size):
    X = []
    for image in images:
        height, width = image.shape[:2]
        if height < width:
            offset = int((width - height) / 2)
            image = image[:, offset:(height + offset)]
        elif height > width:
            offset = int((height - width) / 2)
            image = image[offset:(width + offset), :]
        assert image.shape[0] == image.shape[1], "Cropping didnt work, shape: %s" % image.shape[:2]
        X.append(image)

    return [cv2.resize(img, size) for img in X]


if __name__ == '__main__':
    for dataset in DATA_URLS:
        for url in DATA_URLS[dataset]:
            download(url)