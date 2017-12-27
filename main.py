import logging
import sys
from libs.gtsrb import GTSRB
from libs.stl10 import STL10
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

    # knn = KNN(x_train, y_train, x_test, y_test, n_neighbors=n)
    # knn.start()

    for kernel in ['linear', 'rbf', 'poly']:
        for C in [1, 5, 10, 100]:
            svm = SVM(x_train, y_train, x_test, y_test, kernel=kernel, C=C)
            svm.start()

    # lda = LDA(x_train, y_train, x_test, y_test, tol=tol)
    # lda.start()
    #
    # rs = RandomForest(x_train, y_train, x_test, y_test, n_estimators=n)
    # rs.start()
