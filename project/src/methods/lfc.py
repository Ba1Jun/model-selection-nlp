import warnings

import numpy as np
from numba import njit

# code for the Label-Feature Correlation (LFC) score in A linearized framework and a new benchmark for model selection for fine-tuning


def getLFC(f, y):
    thetaF = np.dot(f, f.T)
    thetaF -= np.mean(thetaF)
    lsm = (y[:, None] == y[None, :]).astype(np.float32) * 2 - 1 # label similariy matrix
    lsm -= np.mean(lsm)
    return np.sum(thetaF * lsm) / (np.linalg.norm(thetaF, ord=2) * np.linalg.norm(lsm, ord=2))


class LFC(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: TransRate score (how well f can fit y directly)
        """
        
        return getLFC(f, y)

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