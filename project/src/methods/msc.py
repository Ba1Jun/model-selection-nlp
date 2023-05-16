import sys
import numpy as np
sys.path.append('/home/baijun/workspace/project/model_selection_nlp/project/src/')
from utils.data import sub_dataset_sampling
from sklearn.metrics import silhouette_score


class MSC(object):
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
        # max_num_data = int(self.args.method.split("-")[1])
        # if f.shape[0] > max_num_data:
        #     f, y = sub_dataset_sampling(f, y, max_num_data, self.args.seed)
        return silhouette_score(f, y, metric="cosine")