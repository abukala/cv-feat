from .common import DATA_PATH, resize
import os
import cv2
import csv

train_dir = DATA_PATH / 'GTSRB' / 'Final_Training' / 'Images'
test_dir = DATA_PATH / 'GTSRB' / 'Final_Test' / 'Images'
gt_file_path = DATA_PATH / 'GT-final_test.csv'
size = (32, 32)
download_url = [
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip",
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip",
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"
]


def load_training_data():
    x_train, y_train = [], []
    for (dir_path, dir_names, files) in os.walk(train_dir):
        dir_name = os.path.basename(dir_path)
        if not dir_name.isdigit():
            continue
        class_id = int(dir_name)
        for file in files:
            if file.endswith('.csv'):
                continue
            img = cv2.imread(os.path.join(dir_path, file))
            x_train.append(img)
            y_train.append(class_id)

    return resize(x_train, size), y_train


def load_test_data():
    x_test, y_test = [], []
    gt_file = csv.reader(gt_file_path.open(), delimiter=';')

    next(gt_file)
    for row in gt_file:
        class_id = int(row[7])
        img = cv2.imread(os.path.join(test_dir, row[0]))
        x_test.append(img)
        y_test.append(class_id)

    return resize(x_test, size), y_test