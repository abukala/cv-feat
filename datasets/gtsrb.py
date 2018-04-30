from .common import DATA_PATH
import os
import csv
import skimage.io
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

train_dir = DATA_PATH / 'GTSRB' / 'Final_Training' / 'Images'
test_dir = DATA_PATH / 'GTSRB' / 'Final_Test' / 'Images'
gt_file_path = DATA_PATH / 'GT-final_test.csv'
size = (32, 32)
download_url = [
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip",
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip",
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"
]


def crop_sq(image):
    height, width = image.shape[:2]
    if height < width:
        offset = int((width - height) / 2)
        image = image[:, offset:(height + offset)]
    elif height > width:
        offset = int((height - width) / 2)
        image = image[offset:(width + offset), :]
    assert image.shape[0] == image.shape[1], "Cropping didnt work, shape: %s" % image.shape[:2]

    return image


def load_training_data():
    path = str(train_dir.absolute())
    x_train, y_train = [], []
    for (dir_path, dir_names, files) in os.walk(path):
        dir_name = os.path.basename(dir_path)
        if not dir_name.isdigit():
            continue
        class_id = int(dir_name)
        for file in files:
            if file.endswith('.csv'):
                continue
            img = skimage.io.imread(os.path.join(dir_path, file))/256
            x_train.append(img)
            y_train.append(class_id)

    return np.array([rgb2gray(resize(crop_sq(img), size)) for img in x_train]), y_train


def load_test_data():
    x_test, y_test = [], []
    gt_file = csv.reader(gt_file_path.open(), delimiter=';')
    path = str(test_dir.absolute())
    next(gt_file)
    for row in gt_file:
        class_id = int(row[7])
        img = skimage.io.imread(os.path.join(path, row[0]))/256
        x_test.append(img)
        y_test.append(class_id)

    return np.array([rgb2gray(resize(crop_sq(img), size)) for img in x_test]), y_test