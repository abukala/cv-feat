from .common import DATA_PATH
import gzip
import struct
import numpy as np

TRAIN_DATA_PATH = DATA_PATH / 'train-images-idx3-ubyte.gz'
TRAIN_LABELS_PATH = DATA_PATH / 'train-labels-idx1-ubyte.gz'
TEST_DATA_PATH = DATA_PATH / 't10k-images-idx3-ubyte.gz'
TEST_LABELS_PATH = DATA_PATH / 't10k-labels-idx1-ubyte.gz'
download_url = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
]

# Img size = (28, 28)


def _load_images(url):
    file = gzip.open(url.open(mode='rb'))
    _ = file.read(4)
    images_count = struct.unpack('>i', file.read(4))[0]
    rows = struct.unpack('>i', file.read(4))[0]
    cols = struct.unpack('>i', file.read(4))[0]
    l = rows * cols
    images = np.array([np.reshape(struct.unpack('>%sB' % l, file.read(l)), (rows, cols)) for _ in range(images_count)], dtype=np.uint8)
    assert len(images) == images_count
    assert images[0].shape == (rows, cols)
    images = np.array([img/256 for img in images])

    return images


def _load_labels(url):
    file = gzip.open(url.open(mode='rb'))
    _ = file.read(4)
    items_count = struct.unpack('>i', file.read(4))[0]
    labels = struct.unpack('>%sB' % items_count, file.read(items_count))
    assert len(labels) == items_count

    return np.array(labels)


def load_training_data():
    return _load_images(TRAIN_DATA_PATH), _load_labels(TRAIN_LABELS_PATH)


def load_test_data():
    return _load_images(TEST_DATA_PATH), _load_labels(TEST_LABELS_PATH)
