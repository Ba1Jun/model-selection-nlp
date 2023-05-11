import warnings

import numpy as np
from numba import njit

# copy from https://github.com/YaojieBao/An-Information-theoretic-Metric-of-Transferability


# def getCov(X):
#     X_mean = X - np.mean(X, axis=0, keepdims=True)
#     cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
#     return cov


# def getHscore(f, Z):
#     #Z=np.argmax(Z, axis=1)
#     Covf = getCov(f)
#     alphabetZ = list(set(Z))
#     g = np.zeros_like(f)
#     for z in alphabetZ:
#         Ef_z = np.mean(f[Z==z, :], axis=0)
#         g[Z==z] = Ef_z
    
#     Covg=getCov(g)
#     score = np.trace(np.dot(np.linalg.pinv(Covf, rcond=1e-15), Covg))
#     return score

def h_score(features: np.ndarray, labels: np.ndarray):
    r"""
    H-score in `An Information-theoretic Approach to Transferability in Task Transfer Learning (ICIP 2019) 
    <http://yangli-feasibility.com/home/media/icip-19.pdf>`_.
    
    The H-Score :math:`\mathcal{H}` can be described as:

    .. math::
        \mathcal{H}=\operatorname{tr}\left(\operatorname{cov}(f)^{-1} \operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)
    
    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector

    Args:
        features (np.ndarray):features extracted by pre-trained model.
        labels (np.ndarray):  groud-truth labels.

    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    f = features
    y = labels

    covf = np.cov(f, rowvar=False)
    C = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar=False)
    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

    return score


class HScore(object):
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
        
        return h_score(f, y)

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