import numpy as np

from sklearn.cluster import KMeans
from scipy.linalg import inv

def run_kmeans(Ninit=10):
    """
    Run the K-means algorithm using the scikit-learn's
    implementation.  
    """
    km = KMeans(init='k-means++', n_clusters=self.K, n_init=Ninit)
    km.fit(self.data)
    return km.cluster_centers_



def negative_log_likelihood(parms,M,data,var,jitter_mode):
    """
    Negative log likelihood
    """
    N = self.data[0]
    D = self.data[1]

    if jitter_mode==0:
        # one jitter number
        assert parms.shape[0]==(D*M + D + 1)
        jitter = parms[-1]
        psis = var + jitter 
    if jitter_mode==1:
        # one jitter vector
        assert parms.shape[1]==(D*M + 2*D)
        jitter = parms[D*M:]
        psis = var + jitter[None,:]
    if jitter_mode==2:
        # N jitter vectors
        assert parms.shape[1]==(D*M + (N+1)*D)
        jitter = parms[D*M:].reshape(N,D)
        psis = var + jitter

    mean = parms[:D]
    lamb = parms[D:D*M].reshape(D,M)
    covs = psis + np.dot(lamb,lamb.T)[None,:]

    nll1 = D * np.log(covs)
    nll2 = (data - means[None,:]) ** 2. / covs

    return np.sum(nll1) + np.sum(nll2)
