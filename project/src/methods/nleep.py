import numpy as np
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA


class NLEEP(object):
    def __init__(self, args):
        self.args = args

    def score(self, X, y, component_ratio=5):

        n = len(y)
        num_classes = len(np.unique(y))
        # PCA: keep 80% energy
        pca_80 = PCA(n_components=0.8)
        pca_80.fit(X)
        X_pca_80 = pca_80.transform(X)

        print(f"After PCA transforming, the dimension is {X_pca_80.shape[1]}")

        # X_pca_80 = X

        # GMM: n_components = component_ratio * class number
        n_components_num = component_ratio * num_classes
        gmm = GaussianMixture(n_components=n_components_num, verbose=1, random_state=self.args.seed).fit(X_pca_80)
        prob = gmm.predict_proba(X_pca_80)  # p(z|x)
        
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