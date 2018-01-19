import os
import pickle
import numpy as np
from .common import DATA_PATH


download_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file



def _load_data(filename):
    data = _unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = _convert_images(raw_images)

    return images, cls


def _unpickle(filename):
    file_path = DATA_PATH / 'cifar-10-batches-py' / filename

    with file_path.open() as file:
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    raw_float = np.array(raw, dtype=np.uint8)
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])

    return images


def load_training_data():
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=np.uint8)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls


def load_test_data():
    images, cls = _load_data(filename="test_batch")

    return images, cls