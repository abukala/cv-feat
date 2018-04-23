from datasets.common import DATA_PATH
from skimage.io import imread
import bz2

FIRST_BATCH = DATA_PATH / 'colorferet' / 'dvd2' / 'gray_feret_cd1' / 'data' / 'images'
SECOND_BATCH = DATA_PATH / 'colorferet' / 'dvd2' / 'gray_feret_cd2' / 'data' / 'images'

if __name__ == '__main__':
    path = FIRST_BATCH
    img = []
    cls = []
    for file in path.iterdir():
        i = imread(bz2.BZ2File(file.open(mode='rb')))
        i = i/255
        img.append(i)
        cls.append(int(file.name[:5]))
        break
