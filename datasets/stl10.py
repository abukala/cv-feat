import numpy as np
from .common import DATA_PATH

stl10_path = DATA_PATH / 'stl10_binary'
download_url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"


def _load_images(url):
    with open(url, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def load_training_data():
    images = _load_images(stl10_path / 'train_X.bin')
    with open(stl10_path / 'train_y.bin') as f:
        clf = np.fromfile(f, dtype=np.uint8)

    return images, clf


def load_test_data():
    images = _load_images(stl10_path / 'test_X.bin')
    with open(stl10_path / 'test_y.bin') as f:
        clf = np.fromfile(f, dtype=np.uint8)

    return images, clf
