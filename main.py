import logging
import sys
from libs.gtsrb import GTSRB
from libs.stl10 import STL10
from libs.mnist import MNIST
from libs.classifiers import KNN, SVM, LDA, RandomForest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        _, dataset, method = sys.argv
    elif len(sys.argv) == 2:
        _, dataset = sys.argv
        method = None
    else:
        logger.error("Insufficient args, exiting...")
        sys.exit()

    if dataset == 'gtsrb':
        img_set = GTSRB()
    elif dataset == 'stl10':
        img_set = STL10()
    elif dataset == 'mnist':
        img_set = MNIST()
    else:
        sys.exit()

    if method == 'sift':
        (x_train, y_train), (x_test, y_test) = img_set.get_sift()
    elif method == 'surf':
        (x_train, y_train), (x_test, y_test) = img_set.get_surf()
    elif method == 'hog':
        (x_train, y_train), (x_test, y_test) = img_set.get_hog()
    else:
        sys.exit()

    knn = KNN(x_train, y_train, x_test, y_test)
    knn.start()

    svm = SVM(x_train, y_train, x_test, y_test)
    svm.start()

    lda = LDA(x_train, y_train, x_test, y_test)
    lda.start()

    rs = RandomForest(x_train, y_train, x_test, y_test)
    rs.start()
