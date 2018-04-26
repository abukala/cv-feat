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
    img = []
    cls = []
    img = [imread(bz2.BZ2File(file.open(mode='rb')))/255 for file in path.iterdir()]
    cls = [int(file.name[:]) for file in path.iterdir()]
    return img, cls

@profile
def load_data(train_ratio=0.7):
    img, cls = _load_batch(FIRST_BATCH)
    print('first batch loaded')
    img2, cls2 = _load_batch(SECOND_BATCH)
    print('second batch loaded')
    img.extend(img2)
    cls.extend(cls2)
    print('lists extended')
    assert len(img) == len(cls)
    choices = np.arange(len(img))
    split = int(len(choices)*train_ratio)
    np.random.seed(0)
    np.random.shuffle(choices)
    print('choices shuffled')
    cls = np.array(cls)
    print('classes to numpy')
    X_train, y_train = img[choices[:split]], cls[choices[:split]]
    print('train set variables')
    X_test, y_test = img[choices[split:]], cls[choices[split:]]
    print('test set variables')
    return (X_train, y_train), (X_test, y_test)


def load_training_data():
    (X, y), (_, _) = load_data()
    return X, y


def load_test_data():
    (_, _), (X, y) = load_data()
    return X, y