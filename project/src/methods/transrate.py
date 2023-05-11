import warnings

import numpy as np
from numba import njit

# copy from https://proceedings.mlr.press/v162/huang22d/huang22d.pdf


def coding_rate(Z, eps=1e-4): 
    n, d = Z.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))
    return 0.5 * rate


def transrate(Z, y, eps=1e-4): 
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps)
    RZY = 0.
    K = int(y.max() + 1)
    for i in range(K):
        RZY += coding_rate(Z[(y==i).flatten()], eps)
    return RZ - RZY / K


class TransRate(object):
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
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        
        return transrate(f, y)

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