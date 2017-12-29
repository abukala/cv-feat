import os
import numpy as np
from .base import BaseLoader
import logging
import pickle

logger = logging.getLogger(__name__)

# image shape
HEIGHT = 32
WIDTH = 32
DEPTH = 3

# root path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# path to data root
ROOT_DATA_DIR = os.path.join(ROOT_PATH, 'data')

# path to cifar-10 dataset
DATA_DIR = os.path.join(ROOT_DATA_DIR, 'cifar-10-batches-py')

# url of the binary data
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class CIFAR10(BaseLoader):
    def __init__(self, **kwargs):
        self.download_and_extract(DATA_URL, ROOT_DATA_DIR)
        BaseLoader.__init__(self, size=(HEIGHT, WIDTH), **kwargs)

    def load_data(self):
        shape = (WIDTH, HEIGHT, DEPTH)
        train_data = []
        for num in range(1, 6):
            with open(os.path.join(DATA_DIR, 'data_batch_%s' % num), 'rb') as f:
                train_data.append(pickle.load(f, encoding='bytes'))

        x_train = []
        y_train = []
        for batch in train_data:
            for img, label in zip(batch[b'data'], batch[b'labels']):
                x_train.append(np.reshape(img, shape))
                y_train.append(label)

        assert len(x_train) == len(y_train)
        assert x_train[0].shape == shape

        with open(os.path.join(DATA_DIR, 'test_batch'), 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')

        x_test = [np.reshape(img, shape) for img in test_data[b'data']]
        y_test = [label for label in test_data[b'labels']]

        assert len(x_test) == len(y_test)
        assert x_test[0].shape == shape

        return (x_train, y_train), (x_test, y_test)

