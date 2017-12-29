import os
import numpy as np
from .base import BaseLoader
import logging
import gzip
import struct

logger = logging.getLogger(__name__)

# image shape
HEIGHT = 28
WIDTH = 28
DEPTH = 3

# root path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# path to data root
ROOT_DATA_DIR = os.path.join(ROOT_PATH, 'data')

# url of the binary data
TRAIN_DATA_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
TEST_DATA_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

# path to the binary train file with image data
TRAIN_DATA_PATH = os.path.join(ROOT_DATA_DIR, 'train-images-idx3-ubyte.gz')

# path to the binary train file with labels
TRAIN_LABELS_PATH = os.path.join(ROOT_DATA_DIR, 'train-labels-idx1-ubyte.gz')

# path to the binary test file with image data
TEST_DATA_PATH = os.path.join(ROOT_DATA_DIR, 't10k-images-idx3-ubyte.gz')

# path to the binary test file with labels
TEST_LABELS_PATH = os.path.join(ROOT_DATA_DIR, 't10k-labels-idx1-ubyte.gz')


class MNIST(BaseLoader):
    def __init__(self, **kwargs):
        self.download_and_extract(TRAIN_DATA_URL, ROOT_DATA_DIR)
        self.download_and_extract(TRAIN_LABELS_URL, ROOT_DATA_DIR)
        self.download_and_extract(TEST_DATA_URL, ROOT_DATA_DIR)
        self.download_and_extract(TEST_LABELS_URL, ROOT_DATA_DIR)
        BaseLoader.__init__(self, size=(HEIGHT, WIDTH), **kwargs)

    def load_data(self):
        x_train = self.load_images(TRAIN_DATA_PATH)
        x_test = self.load_images(TEST_DATA_PATH)
        y_train = self.load_labels(TRAIN_LABELS_PATH)
        y_test = self.load_labels(TEST_LABELS_PATH)

        return (x_train, y_train), (x_test, y_test)

    def load_images(self, url):
        file = gzip.open(url, 'rb')
        _ = file.read(4)
        images_count = struct.unpack('>i', file.read(4))[0]
        rows = struct.unpack('>i', file.read(4))[0]
        cols = struct.unpack('>i', file.read(4))[0]
        assert rows == WIDTH
        assert cols == HEIGHT
        l = rows*cols
        images = [np.reshape(struct.unpack('>%sB' % l, file.read(l)), (rows, cols)) for num in range(images_count)]
        assert len(images) == images_count
        assert images[0].shape == (rows, cols)

        return images

    def load_labels(self, url):
        file = gzip.open(url, 'rb')
        _ = file.read(4)
        items_count = struct.unpack('>i', file.read(4))[0]
        labels = struct.unpack('>%sB' % items_count, file.read(items_count))
        assert len(labels) == items_count

        return labels
