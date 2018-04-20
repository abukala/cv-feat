import numpy as np
from .common import DATA_PATH
from skimage.color import rgb2gray

stl10_path = DATA_PATH / 'stl10_binary'
download_url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

# Img size = (96, 96, 3)


def _load_images(url):
    with url.open(mode='rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        images = np.array([rgb2gray(img) for img in images])
        return images


def load_training_data():
    images = _load_images(stl10_path / 'train_X.bin')
    path = stl10_path / 'train_y.bin'
    with path.open() as f:
        clf = np.fromfile(f, dtype=np.uint8)

    return images, clf


def load_test_data():
    images = _load_images(stl10_path / 'test_X.bin')
    path = stl10_path / 'test_y.bin'
    with path.open() as f:
        clf = np.fromfile(f, dtype=np.uint8)

    return images, clf
