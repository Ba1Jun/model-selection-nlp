import warnings
import scipy
import sklearn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class PARC(object):
    def __init__(self, args):
        self.args = args
    
    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions,1)]

    def score(self, features, labels):
        """PARC score from https://github.com/dbolya/parc/blob/main/methods.py

        Args:
            features (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """        
        scaler = sklearn.preprocessing.StandardScaler()
        features  = scaler.fit_transform(features)
        num_classes = len(set(labels))
        labels = np.eye(num_classes)[labels]

        dist = str(self.args.method.split("-")[1])
        if dist == 'corr':
            rdm1 = 1 - np.corrcoef(features)
            rdm2 = 1 - np.corrcoef(labels)
        elif dist == 'cos':
            rdm1 = 1 - cosine_similarity(features)
            rdm2 = 1 - cosine_similarity(labels)
        elif dist == "l2":
            rdm1 = euclidean_distances(features)
            rdm2 = euclidean_distances(labels)
        elif dist == "dot":
            rdm1 = -np.dot(features, features.T)
            rdm2 = -np.dot(labels, labels.T)
        
        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)
        
        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100