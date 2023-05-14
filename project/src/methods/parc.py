import warnings
import scipy
import sklearn
import numpy as np


class PARC(object):
    def __init__(self, args):
        self.args = args
    
    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions,1)]

    def score(self, features, labels):
        scaler = sklearn.preprocessing.StandardScaler()
        features  = scaler.fit_transform(features)
        num_classes = len(set(labels))
        labels = np.eye(num_classes)[labels]

        rdm1 = 1 - np.corrcoef(features)
        rdm2 = 1 - np.corrcoef(labels)
        
        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)
        
        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100