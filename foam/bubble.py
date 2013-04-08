import numpy as np

from sklearn.cluster import KMeans

class SingleFA(object):

    def __init__(self,data,obs_var,
                 latent_dim = 2,
                 grad_descent=False)

        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        self.M = latent_dim
        self.K = 1

        self.data = data
        self.obs_var = var

        self.initialize()
        #self.optimize()

    def initialize(self):

        self.mean = self.run_kmeans()
        self.lamb = np.zeros((self.D,self.M))
        self.jits = np.zeros(self.D)

        self.make_psis()
        self.make_covs()

    def run_kmeans(Ninit=10):
        """
        Run the K-means algorithm using the scikit-learn's
        implementation.  
        """
        km = KMeans(init='k-means++', n_clusters=self.K, n_init=Ninit)
        km.fit(self.data)
        return km.cluster_centers_

    def make_psis(self):
        # N x D
        return self.obs_var + self.jits[None,:]

    def make_covs(self):

        # DxD
        lamlamT = np.dot(self.lamb,self.lamb.T)
        print lamlamT.shape
        
