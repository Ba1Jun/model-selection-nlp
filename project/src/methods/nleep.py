import numpy as np
from sklearn.mixture import GaussianMixture
import sys
sys.path.append('/home/baijun/workspace/project/model_selection_nlp/project/src/')
from utils.data import sub_dataset_sampling


class NLEEP(object):
    def __init__(self, args):
        self.args = args

    def score(self, X, y, component_ratio=5):

        max_num_data = int(self.args.method.split("-")[1])
        if X.shape[0] > max_num_data:
            X, y = sub_dataset_sampling(X, y, max_num_data, self.args.seed)

        n = len(y)
        num_classes = len(np.unique(y))

        # GMM: n_components = component_ratio * class number
        n_components_num = component_ratio * num_classes
        gmm = GaussianMixture(n_components=n_components_num, verbose=1, random_state=self.args.seed).fit(X)
        prob = gmm.predict_proba(X)  # p(z|x)
        
        # NLEEP
        pyz = np.zeros((num_classes, n_components_num))
        for y_ in range(num_classes):
            indices = np.where(y == y_)[0]
            filter_ = np.take(prob, indices, axis=0) 
            pyz[y_] = np.sum(filter_, axis=0) / n   
        pz = np.sum(pyz, axis=0)    
        py_z = pyz / pz             
        py_x = np.dot(prob, py_z.T) 

        # nleep_score
        nleep_score = np.sum(py_x[np.arange(n), y]) / n
        return nleep_score