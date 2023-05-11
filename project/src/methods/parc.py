import warnings
import scipy
import sklearn
import numpy as np


def get_lowertri(rdm):
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions,1)]


def get_parc_correlation(features, labels):
    scaler = sklearn.preprocessing.StandardScaler()
    features  = scaler.fit_transform(features)
    num_classes = len(set(labels))
    labels = np.eye(num_classes)[labels]

    rdm1 = 1 - np.corrcoef(features)
    rdm2 = 1 - np.corrcoef(labels)
    
    lt_rdm1 = get_lowertri(rdm1)
    lt_rdm2 = get_lowertri(rdm2)
    
    return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100


class PARC(object):
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
        
        return get_parc_correlation(f, y)

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