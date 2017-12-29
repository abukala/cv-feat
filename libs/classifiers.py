from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import threading
import logging

logger = logging.getLogger(__name__)


class SVM(threading.Thread):
    def __init__(self, data, kernel='linear', C=0.1, tol=0.01):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.score = None
        threading.Thread.__init__(self)

    def run(self):
        clf = SVC(kernel=self.kernel, C=self.C, tol=self.tol)
        clf = clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        self.score = accuracy_score(self.y_test, y_pred)
        logger.info("SVM(C: %s, %s) score: %s" % (self.C, self.kernel, self.score))


class RandomForest(threading.Thread):
    def __init__(self, data, n_estimators=100):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.n_estimators = n_estimators
        self.score = None
        threading.Thread.__init__(self)

    def run(self):
        clf = RandomForestClassifier(n_estimators=self.n_estimators)
        clf = clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        self.score = accuracy_score(self.y_test, y_pred)
        logger.info("Random Forest(%s estimators) score: %s" % (self.n_estimators, self.score))


class LDA(threading.Thread):
    def __init__(self, data, tol=0.1):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.tol = tol
        self.score = None
        threading.Thread.__init__(self)

    def run(self):
        clf = LinearDiscriminantAnalysis(tol=self.tol)
        clf = clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        self.score = accuracy_score(self.y_test, y_pred)
        logger.info("LDA(tol: %s) score: %s" % (self.tol, self.score))


class KNN(threading.Thread):
    def __init__(self, data, n_neighbors=10):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.n_neighbors = n_neighbors
        self.score = None
        threading.Thread.__init__(self)

    def run(self):
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        clf = clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        self.score = accuracy_score(self.y_test, y_pred)
        logger.info("KNN(%s neighbors) score: %s" % (self.n_neighbors, self.score))