import numpy as np

# code for the Label-Feature Correlation (LFC) score in A linearized framework and a new benchmark for model selection for fine-tuning


class LFC(object):
    def __init__(self, args):
        self.args = args

    def score(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: TransRate score (how well f can fit y directly)
        """
        thetaF = np.dot(f, f.T)
        thetaF -= np.mean(thetaF)
        lsm = (y[:, None] == y[None, :]).astype(np.float32) * 2 - 1 # label similariy matrix
        lsm -= np.mean(lsm)
        return np.sum(thetaF * lsm) / (np.linalg.norm(thetaF, ord=2) * np.linalg.norm(lsm, ord=2))