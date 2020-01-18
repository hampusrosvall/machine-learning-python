import numpy as np

class NearestCentroidClassifier:

    def __init__(self):
        self.n_classes = None
        self.n_features = None
        self.mean_vectors = None

    def fit(self, X_train, y_train):
        self.n_classes = len(np.unique(y_train))
        self.n_features = len(X_train[0, :])
        self.mean_vectors = np.zeros((self.n_classes, self.n_features))
        for cl in np.unique(y_train):
            self.mean_vectors[cl, :] = np.mean(X_train[y_train == cl, :], axis = 0)

    def predict(self, X_test):
        predictions = []
        for test_example in X_test:
            predictions.append(np.argmin([np.linalg.norm(test_example - mean_x) for mean_x in self.mean_vectors]))

        return np.array(predictions)

if __name__ == '__main__':
    NCC = NearestCentroidClassifier()