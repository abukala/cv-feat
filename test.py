from datasets.feret import load_data

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)