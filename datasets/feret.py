from .common import DATA_PATH
from skimage.io import imread
import bz2
import numpy as np
from memory_profiler import profile

FIRST_BATCH = DATA_PATH / 'colorferet' / 'dvd2' / 'gray_feret_cd1' / 'data' / 'images'
SECOND_BATCH = DATA_PATH / 'colorferet' / 'dvd2' / 'gray_feret_cd2' / 'data' / 'images'

params = {
    'hog': {
        'pixels_per_cell': (24, 24),
        'cells_per_block': (1, 1)
    }
}

# Img shape = (256, 384)


def _load_batch(path):
    img = np.array([imread(bz2.BZ2File(file.open(mode='rb')))/255 for file in path.iterdir()])
    cls = np.array([int(file.name[:5]) for file in path.iterdir()])
    return img, cls


@profile
def load_data(train_ratio=0.7):
    img = []
    cls = []
    for file in FIRST_BATCH.iterdir():
        img.append(imread(file)/255)
        cls.append(int(file.name[:5]))
    for file in SECOND_BATCH.iterdir():
        img.append(imread(file)/255)
        cls.append(int(file.name[:5]))
    assert len(img) == len(cls)
    choices = np.arange(len(img))
    split = int(len(choices)*train_ratio)
    np.random.seed(0)
    np.random.shuffle(choices)
    X_train = [img[choice] for choice in choices[:split]]
    y_train = [cls[choice] for choice in choices[:split]]
    X_test = [img[choice] for choice in choices[split:]]
    y_test = [cls[choice] for choice in choices[split:]]
    return (X_train, y_train), (X_test, y_test)


def load_training_data():
    (X, y), (_, _) = load_data()
    return X, y


def load_test_data():
    (_, _), (X, y) = load_data()
    return X, y