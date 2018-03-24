from .common import DATA_PATH
from skimage.io import imread
import bz2
import numpy as np

FIRST_BATCH = DATA_PATH / 'colorferet' / 'dvd2' / 'gray_feret_cd1' / 'data' / 'images'
SECOND_BATCH = DATA_PATH / 'colorferet' / 'dvd2' / 'gray_feret_cd2' / 'data' / 'images'


def _load_batch(path, subset=None):
    img = []
    cls = []
    for file in path.iterdir():
        if subset:
            if file.name[5:7] != subset:
                continue
        img.append(imread(bz2.BZ2File(file.open(mode='rb'))))
        cls.append(int(file.name[:5]))
    return np.array(img), np.array(cls)


def load_data(subset=None, train_ratio=0.6):
    img, cls = _load_batch(FIRST_BATCH, subset)
    img2, cls2 = _load_batch(SECOND_BATCH, subset)
    img.extend(img2)
    cls.extend(cls2)
    assert len(img) == len(cls)
    choices = np.arange(len(img))
    split = int(len(choices)*train_ratio)
    np.random.shuffle(choices)
    X_train, y_train = img[choices[:split]], cls[choices[:split]]
    X_test, y_test = img[choices[split:]], cls[choices[split:]]

    return (X_train, y_train), (X_test, y_test)