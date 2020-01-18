import numpy as np
from collections import Counter
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:

    def __init__(self):
        self.n_classes = None
        self.probability_table = None
        self.priors = None

    def fit(self, X_train, y_train):
        # initailize priors
        self.priors = np.array([len(y_train[y_train == cl]) for cl in np.unique(y_train)]) / len(y_train)

        # extract dimensions
        self.n_classes = len(np.unique(y_train))
        scale = int(np.max(X_train))
        n_features = len(X_train[0, :])

        # initialize probability table
        self.probability_table = np.ones((self.n_classes, scale + 1, n_features))

        # build probability table
        for idx, x in enumerate(X_train):
            for pxl_idx, pxl_val in enumerate(x):
                self.probability_table[y_train[idx], int(pxl_val), pxl_idx] += 1

        # normalize
        n_per_class = Counter(y_train)
        for key, val in n_per_class.items():
            self.probability_table[key, :, :] /= val

    def predict(self, X_test):
        predictions = []

        # calculate posteriors
        for x in X_test:
            probs = []
            for cl in range(self.n_classes):
                prob_cl = 1
                for idx, pxl in enumerate(x):
                    prob_cl *= self.probability_table[cl, int(pxl), idx]
                probs.append(prob_cl)

            # multiply with prior
            probs *=self.priors

            predictions.append(np.argmax(probs))

        return predictions

if __name__ == '__main__':
    data, target = datasets.load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

    nbc = NaiveBayesClassifier()
    nbc.fit(X_train, y_train)
    y_pred = nbc.predict(X_test)

    print("Classification report SKLearn NBC:\n%s\n"
          % (metrics.classification_report(y_test, y_pred)))
    print("Confusion matrix SKLearn NBC:\n%s" % metrics.confusion_matrix(y_test, y_pred))