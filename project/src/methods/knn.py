import warnings
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class kNN(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray,
                  val_features: np.ndarray, val_labels: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: TransRate score (how well f can fit y directly)
        """

        model = KNeighborsClassifier(n_neighbors=1).fit(train_features, train_labels)
        return (model.predict(val_features) == val_labels).mean()

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        return np.argmax(logits, axis=-1)