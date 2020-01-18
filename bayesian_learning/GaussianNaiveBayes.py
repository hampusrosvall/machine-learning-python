import numpy as np
from collections import Counter
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split

class GaussianNaiveBayes:

    def __init__(self):
        self.n_classes = None
        self.mean_table = None
        self.variance_table = None
        self.priors = None

    def fit(self, X_train, y_train):
        # initialize priors
        self.priors = np.array([len(y_train[y_train == cl]) for cl in np.unique(y_train)]) / len(y_train)

        # extract dimensions
        self.n_classes = len(np.unique(y_train))
        n_features = len(X_train[0, :])

        # initialize tables
        self.mean_table = np.zeros((self.n_classes, n_features))
        self.variance_table = self.mean_table.copy()

        # build mean and variance table
        for cl in np.unique(y_train):
            self.mean_table[cl, :] = np.mean(X_train[y_train == cl], axis = 0)
            self.variance_table[cl, :] = np.var(X_train[y_train == cl], axis = 0)

        # fix zero variances
        self.variance_table[self.variance_table == 0] = 0.01
        self.mean_table[self.mean_table == 0] = 0.01


    def predict_old(self, X_test):
        predictions = []

        for x in X_test:
            cl_pr = []
            for cl in range(self.n_classes):
                pot = np.square(x - self.mean_table[cl, :])
                pot = np.divide(pot, 2 * self.variance_table[cl, :])
                probs = np.divide(np.exp(-pot), np.sqrt(2 * np.pi * self.variance_table))
                cl_pr.append(np.prod(probs))
            predictions.append(np.argmax(cl_pr * self.priors))

        return predictions

    def predict(self, X_test):
        predictions = []

        for x in X_test:
            pot = np.square(x - self.mean_table) / (2 * self.variance_table)
            probs = np.exp(-pot) / np.sqrt(2 * np.pi * self.variance_table)
            probs[probs == 0] = 1e-10
            log_probs = np.log(probs)
            probs_pr = np.sum(log_probs, axis = 1) + np.log(self.priors)
            predictions.append(np.argmax(probs_pr))

        return np.array(predictions)


if __name__ == '__main__':
    data, target = datasets.load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

    nbc = GaussianNaiveBayes()
    nbc.fit(X_train, y_train)
    y_pred = nbc.predict(X_test)

    print("Classification report SKLearn NBC:\n%s\n"
          % (metrics.classification_report(y_test, y_pred)))
    print("Confusion matrix SKLearn NBC:\n%s" % metrics.confusion_matrix(y_test, y_pred))

