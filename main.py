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
    if len(sys.argv) == 4:
        _, dataset, method, scan = sys.argv
    elif len(sys.argv) == 3:
        _, dataset, method = sys.argv
        scan = False
    elif len(sys.argv) == 2:
        _, dataset = sys.argv
        method = None
        scan = False
    else:
        logger.error("Insufficient args, exiting...")
        sys.exit()

    if dataset == 'gtsrb':
        img_set = GTSRB()
    elif dataset == 'stl10':
        img_set = STL10()
    else:
        sys.exit()

    if not scan:
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
    else:
        if method == 'sift':
            for k in [200, 500, 1000, 2500, 5000]:
                for multi in [10, 25, 50, 100]:
                    for max_iter in [10, 50, 100, 200, 500]:
                        for n_init in [1, 5, 10]:
                            (x_train, y_train), (x_test, y_test) = img_set.get_sift(k, multi, n_init, max_iter)

                            logger.info("SIFT-grid || k: %s || multi: %s || max_iter: %s || n_init: %s" % (k, multi, max_iter, n_init))

                            knn = KNN(x_train, y_train, x_test, y_test)
                            knn.start()

                            svm = SVM(x_train, y_train, x_test, y_test)
                            svm.start()

                            lda = LDA(x_train, y_train, x_test, y_test)
                            lda.start()

                            rs = RandomForest(x_train, y_train, x_test, y_test)
                            rs.start()

                            knn.join()
                            svm.join()
                            lda.join()
                            rs.join()

        elif method == 'surf':
            for k in [200, 500, 1000, 2500, 5000]:
                for multi in [10, 25, 50, 100]:
                    for max_iter in [10, 50, 100, 200, 500]:
                        for n_init in [1, 5, 10]:
                            (x_train, y_train), (x_test, y_test) = img_set.get_sift(k, multi, n_init, max_iter)

                            logger.info("SURF-grid || k: %s || multi: %s || max_iter: %s || n_init: %s" % (k, multi, max_iter, n_init))

                            knn = KNN(x_train, y_train, x_test, y_test)
                            knn.start()

                            svm = SVM(x_train, y_train, x_test, y_test)
                            svm.start()

                            lda = LDA(x_train, y_train, x_test, y_test)
                            lda.start()

                            rs = RandomForest(x_train, y_train, x_test, y_test)
                            rs.start()

                            knn.join()
                            svm.join()
                            lda.join()
                            rs.join()