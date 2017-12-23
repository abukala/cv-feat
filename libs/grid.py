import threading
from .classifiers import KNN, SVM, LDA, RandomForest
import logging

logger = logging.getLogger(__name__)


class GridSearchSIFT(threading.Thread):
    def __init__(self, img_set, k, multi, n_init, max_iter):
        self.img_set = img_set
        self.k = k
        self.multi = multi
        self.n_init = n_init
        self.max_iter = max_iter
        threading.Thread.__init__(self)

    def run(self):
        (x_train, y_train), (x_test, y_test) = self.img_set.get_sift(k=self.k, multi=self.multi,
                                                                     n_init=self.n_init, max_iter=self.max_iter)

        knn = KNN(x_train, y_train, x_test, y_test)
        knn.start()

        svm = SVM(x_train, y_train, x_test, y_test)
        svm.start()

        lda = LDA(x_train, y_train, x_test, y_test)
        lda.start()

        rf = RandomForest(x_train, y_train, x_test, y_test)
        rf.start()

        knn.join()
        logger.info("SIFT-KNN score: %s || k: %s - multi: %s - n_init: %s - max_iter: %s" % (knn.score, self.k, self.multi, self.n_init, self.max_iter))
        svm.join()
        logger.info("SIFT-SVM score: %s || k: %s - multi: %s - n_init: %s - max_iter: %s" % (svm.score, self.k, self.multi, self.n_init, self.max_iter))
        lda.join()
        logger.info("SIFT-LDA score: %s || k: %s - multi: %s - n_init: %s - max_iter: %s" % (lda.score, self.k, self.multi, self.n_init, self.max_iter))
        rf.join()
        logger.info("SIFT-RS score: %s || k: %s - multi: %s - n_init: %s - max_iter: %s" % (rf.score, self.k, self.multi, self.n_init, self.max_iter))


class GridSearchSURF(threading.Thread):
    def __init__(self, img_set, k, multi, n_init, max_iter):
        self.img_set = img_set
        self.k = k
        self.multi = multi
        self.n_init = n_init
        self.max_iter = max_iter
        threading.Thread.__init__(self)

    def run(self):
        (x_train, y_train), (x_test, y_test) = self.img_set.get_surf(k=self.k, multi=self.multi,
                                                                     n_init=self.n_init, max_iter=self.max_iter)

        knn = KNN(x_train, y_train, x_test, y_test)
        knn.start()

        svm = SVM(x_train, y_train, x_test, y_test)
        svm.start()

        lda = LDA(x_train, y_train, x_test, y_test)
        lda.start()

        rf = RandomForest(x_train, y_train, x_test, y_test)
        rf.start()

        knn.join()
        logger.info("SURF-KNN score: %s || k: %s - multi: %s - n_init: %s - max_iter: %s" % (knn.score, self.k, self.multi, self.n_init, self.max_iter))
        svm.join()
        logger.info("SURF-SVM score: %s || k: %s - multi: %s - n_init: %s - max_iter: %s" % (svm.score, self.k, self.multi, self.n_init, self.max_iter))
        lda.join()
        logger.info("SURF-LDA score: %s || k: %s - multi: %s - n_init: %s - max_iter: %s" % (lda.score, self.k, self.multi, self.n_init, self.max_iter))
        rf.join()
        logger.info("SURF-RS score: %s || k: %s - multi: %s - n_init: %s - max_iter: %s" % (rf.score, self.k, self.multi, self.n_init, self.max_iter))