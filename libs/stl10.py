import os
import numpy as np
from .base import BaseLoader
import logging

logger = logging.getLogger(__name__)

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# root path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# path to data root
ROOT_DATA_DIR = os.path.join(ROOT_PATH, 'data')

# path to the stl10 dataset
DATA_DIR = os.path.join(ROOT_DATA_DIR, 'stl10_binary')

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_X.bin')

# path to the binary train file with labels
TRAIN_LABEL_PATH = os.path.join(DATA_DIR, 'train_y.bin')

# path to the binary test file with image data
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_X.bin')

# path to the binary test file with labels
TEST_LABEL_PATH = os.path.join(DATA_DIR, 'test_y.bin')


class STL10(BaseLoader):
    def __init__(self, **kwargs):
        self.download_and_extract(DATA_URL, ROOT_DATA_DIR)
        BaseLoader.__init__(self, **kwargs)

    def load_data(self):
        x_train = self.read_all_images(TRAIN_DATA_PATH)
        x_test = self.read_all_images(TEST_DATA_PATH)
        y_train = self.read_labels(TRAIN_LABEL_PATH)
        y_test = self.read_labels(TEST_LABEL_PATH)

        return (x_train, y_train), (x_test, y_test)

    def read_labels(self, path_to_labels):
        """
        :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
        :return: an array containing the labels
        """
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(self, path_to_data):
        """
        :param path_to_data: the file containing the binary images from the STL-10 dataset
        :return: an array containing all the images
        """

        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)

            # We force the data into 3x96x96 chunks, since the
            # images are stored in "column-major order", meaning
            # that "the first 96*96 values are the red channel,
            # the next 96*96 are green, and the last are blue."
            # The -1 is since the size of the pictures depends
            # on the input file, and this way numpy determines
            # the size on its own.

            images = np.reshape(everything, (-1, 3, 96, 96))

            # Now transpose the images into a standard image format
            # readable by, for example, matplotlib.imshow
            # You might want to comment this line or reverse the shuffle
            # if you will use a learning algorithm like CNN, since they like
            # their channels separated.
            images = np.transpose(images, (0, 3, 2, 1))
            return images
