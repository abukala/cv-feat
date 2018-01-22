from datasets import gtsrb, mnist, stl10

(X_train, y_train), (X_test, y_test) = gtsrb.load_training_data(), gtsrb.load_test_data()
(X_train, y_train), (X_test, y_test) = stl10.load_training_data(), stl10.load_test_data()
(X_train, y_train), (X_test, y_test) = mnist.load_training_data(), mnist.load_test_data()
