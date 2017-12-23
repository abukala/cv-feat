import cv2
import csv
import os
from .base import BaseLoader

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
GTSRB_PATH = os.path.join(ROOT_PATH, 'data', 'GTSRB')

GTSRB_TRAIN_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
GTSRB_TEST_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip'
GTSRB_CL_ID_URL = 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip'


class GTSRB(BaseLoader):
    def __init__(self, **kwargs):
        self.download_and_extract(GTSRB_TRAIN_URL, './data')
        self.download_and_extract(GTSRB_TEST_URL, './data')
        self.download_and_extract(GTSRB_CL_ID_URL, './data', os.path.join(GTSRB_PATH, 'Final_Test', 'Images'))
        BaseLoader.__init__(self, **kwargs)

    def load_data(self):
        train_dir = os.path.join(GTSRB_PATH, 'Final_Training', 'Images')
        x_train, y_train, x_test, y_test = [], [], [], []

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

        test_dir = os.path.join(GTSRB_PATH, 'Final_Test', 'Images')
        gt_file = csv.reader(open(os.path.join(test_dir, 'GT-final_test.csv')), delimiter=';')

        next(gt_file)
        for row in gt_file:
            class_id = int(row[7])
            img = cv2.imread(os.path.join(test_dir, row[0]))
            x_test.append(img)
            y_test.append(class_id)

        return (x_train, y_train), (x_test, y_test)