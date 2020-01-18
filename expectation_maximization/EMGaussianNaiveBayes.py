import numpy as np
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split

class EMGaussianNaiveBayes:

    def __init__(self):
        self.n_classes = None
        self.mean_table = None
        self.variance_table = None
        self.priors = None
        self.epsilon = 1e-8
        self.look_up_table = None

    def fit(self, X_train, n_classes):
        # declare classes
        self.n_classes = n_classes

        # declare priors
        self.priors = np.random.rand(n_classes)

        # declare number of attributes
        n_features = len(X_train[0, :])

        # initialize tables
        self.mean_table = np.random.rand(self.n_classes, n_features)
        self.variance_table = self.mean_table.copy()

        # expectation maximization algorithm to find mean and variances
        prev_mean = self.mean_table.copy() * 1000
        prev_var = self.variance_table.copy() * 1000
        threshold = 1e-8

        while ((np.linalg.norm(self.mean_table - prev_mean, 'fro') > threshold) or
            (np.linalg.norm(self.variance_table - prev_var, 'fro') > threshold)):

            prev_mean = self.mean_table.copy()
            prev_var = self.variance_table.copy()

            r_tot = np.zeros((n_classes, len(X_train)))
            for idx, x in enumerate(X_train):
                exponent = np.square(x - self.mean_table) / (2 * self.variance_table)
                probs = np.exp(-exponent) / np.sqrt(2 * np.pi * self.variance_table)
                probs_pr = np.prod(probs, axis=1) * self.priors
                r_i = probs_pr / np.sum(probs_pr)
                r_tot[:, idx] = r_i

            # M-step
            r_k = np.sum(r_tot, axis = 1)
            self.priors = r_k / len(X_train)

            for cl in range(n_classes):
                acc_mu = np.zeros(n_features)
                acc_cov_matrix = np.zeros((n_features, n_features))
                for i, x_i in enumerate(X_train):
                    acc_mu += r_tot[cl, i] * x_i
                    acc_cov_matrix += r_tot[cl, i] * np.outer(x_i, x_i)

                # updating the mean_table
                self.mean_table[cl, :] = acc_mu / r_k[cl]

                # calculate covariance matrix
                cov_matrix = acc_cov_matrix / r_k[cl] - np.outer(self.mean_table[cl, :], self.mean_table[cl, :])

                cov_matrix += self.epsilon

                # get diagonal elements
                self.variance_table[cl, :] = cov_matrix.diagonal()



        return True

    def predict(self, X_test):
        # produce a confusion matrix

        predictions = []

        for x in X_test:
            pot = np.square(x - self.mean_table) / (2 * self.variance_table)
            probs = np.exp(-pot) / np.sqrt(2 * np.pi * self.variance_table)
            probs_pr = np.prod(probs, axis = 1) * self.priors
            predictions.append(np.argmax(probs_pr))

        return np.array(predictions)


if __name__ == '__main__':
    data, target = datasets.load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    X_train /= 16
    X_test /= 16

    nbc = EMGaussianNaiveBayes()
    nbc.fit(X_train, 10)
    y_pred = nbc.predict(X_train)

    print("Classification report SKLearn NBC:\n%s\n"
          % (metrics.classification_report(y_train, y_pred)))
    print("Confusion matrix SKLearn NBC:\n%s" % metrics.confusion_matrix(y_train, y_pred))