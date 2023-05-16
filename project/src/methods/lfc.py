import numpy as np
import sys
sys.path.append('/home/baijun/workspace/project/model_selection_nlp/project/src/')
from utils.data import sub_dataset_sampling

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
        max_num_data = int(self.args.method.split("-")[1])
        if f.shape[0] > max_num_data:
            f, y = sub_dataset_sampling(f, y, max_num_data, self.args.seed)
        thetaF = np.dot(f, f.T)
        thetaF -= np.mean(thetaF)
        lsm = (y[:, None] == y[None, :]).astype(np.float32) * 2 - 1 # label similariy matrix
        lsm -= np.mean(lsm)
        return np.sum(thetaF * lsm) / (np.linalg.norm(thetaF, ord=2) * np.linalg.norm(lsm, ord=2))