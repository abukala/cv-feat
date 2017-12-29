import logging
import sys
import argparse
from libs.gtsrb import GTSRB
from libs.stl10 import STL10
from libs.mnist import MNIST
from libs.cifar10 import CIFAR10
from libs.classifiers import KNN, SVM, LDA, RandomForest
import pickle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--method')
    parser.add_argument('--process')
    args = parser.parse_args()

    if args.dataset:
        if args.dataset == 'gtsrb':
            img_set = GTSRB()
        elif args.dataset == 'stl10':
            img_set = STL10()
        elif args.dataset == 'mnist':
            img_set = MNIST()
        elif args.dataset == 'cifar10':
            img_set = CIFAR10()
        else:
            sys.exit()

        if args.method == 'sift':
            data = img_set.get_sift()
        elif args.method == 'surf':
            data = img_set.get_surf()
        elif args.method == 'hog':
            data = img_set.get_hog()
        else:
            sys.exit()

        knn = KNN(data)
        knn.start()

        svm = SVM(data)
        svm.start()

        lda = LDA(data)
        lda.start()

        rf = RandomForest(data)
        rf.start()
    else:
        if not args.process:
            results = {}

            gtsrb = GTSRB().start()
            stl = STL10().start()
            mnist = MNIST().start()
            cifar = CIFAR10().start()

            gtsrb.join()
            stl.join()
            mnist.join()
            cifar.join()

            for ds_name in ['gtsrb', 'stl', 'mnist', 'cifar']:
                for method in ['pix', 'hog', 'sift', 'surf']:
                    dataset = eval(ds_name)
                    knn = KNN(getattr(dataset, method)).start()
                    svm = SVM(getattr(dataset, method)).start()
                    lda = LDA(getattr(dataset, method)).start()
                    rf = RandomForest(getattr(dataset, method)).start()

                    for clf_name in ['knn', 'svm', 'lda', 'rf']:
                        clf = eval(clf_name)
                        clf.join()
                        results[ds_name][method][clf_name] = clf.score

            pickle.dump(results, open('results.p', 'wb'))
