import numpy as np

class HMF(object):
    """
    Heteroskedastic Matrix Factorization

    Factorization using the "known" observational variances,
    presented in Tzalmantza & Hogg (2011) and implemented in
    python (without regularization) by Bailey (2012).

    The version here matches Tzalmantza & Hogg, but also includes
    a model for the mean.

    Parameters
    ----------
    data : n_feature by n_sample array of the data
    K    : the number of latent dimensions.
    ivar : Array of inverse variances associated with the data.
    eps  : Regularization strength
    tol  : Percent tolerance criterion for delta negative log likelihood.
    max_iter   : Maximum number of iterations to run EM.
    check_iter : Interval number of EM iterations between convergence checks.

    To Do:
    ------
    - regularization
    - Automatic latent dimensionality determination.
    """
    def __init__(self, data, K, ivar, max_iter=1000, check_iter=5,
                 tol=1.e-4):
        """
        `D` : Feature dimensionality
        `N` : Number of samples
        `K` : Latent dimensionality
        `lambdas` : Latent transformation matrix, shape K x D
        `latents` : Projection of latents, shape
        """
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.K = K 
        self.data = np.atleast_2d(data)
        self.ivar = ivar

        # initial mean
        self.mean = np.sum(self.data * self.ivar, axis=0)
        self.mean /= np.sum(self.ivar, axis=0)
        self.dmm = self.data - self.mean

        # initial lambdas
        self.lambdas, evals = self.svd_pca(self.dmm)
        self.lambdas = self.lambdas[:self.K, :]

        # initialize latents and project
        self.latents = np.zeros((self.K, self.N))
        self._e_step()
        self.project()

        # go
        self.run_em(tol, max_iter, check_iter)
        
    def svd_pca(self, data):
        """
        PCA using a singular value decomposition.
        """
        U, S, eigvec = np.linalg.svd(data)
        eigval = S ** 2. / (data.shape[0] - 1.)
        return eigvec, eigval

    def run_em(self, tol, max_iter, check_iter):
        """
        Use expectation-maximization to infer the model.
        """
        nll = 0.5 * np.sum((self.dmm - self.projections) ** 2. * self.ivar)
        print 'Starting NLL =', nll
        for i in range(max_iter):
            self._e_step()
            self._m_step()
            if np.mod(i, check_iter) == 0:
                new_nll =  0.5 * np.sum((self.dmm - self.projections) ** 2. * self.ivar)
                print 'NLL at step %d is:' % i, new_nll
            if ((nll - new_nll) / nll) < tol:
                print 'Stopping at step %d with NLL:', new_nll
            else:
                nll = new_nll

    def _e_step(self):
        """
        Infer the latent variables. 'a-step' in Tzalmantza & Hogg
        """
        for i in range(self.N):
            ilTl = np.linalg.inv(np.dot(self.lambdas * self.ivar[i], self.lambdas.T))
            lTx = np.dot(self.lambdas, self.dmm[i] * self.ivar[i])
            self.latents[:, i] = np.dot(ilTl, lTx)

    def _m_step(self):
        """
        Learn lambdas, mean. 'a-step' in Tzalmantza & Hogg
        """
        for j in range(self.D):
            ilTl = np.linalg.inv(np.dot(self.latents * self.ivar[:, j], self.latents.T))
            lTx = np.dot(self.latents, self.dmm[:, j] * self.ivar[:, j])
            self.lambdas[:, j] = np.dot(ilTl, lTx)

        self._norm_and_ortho_ize()
        self.project()

        self.mean = np.sum((self.data - self.projections) * self.ivar, axis=0)
        self.mean /= np.sum(self.ivar, axis=0)
        self.dmm = self.data - self.mean

    def _norm_and_ortho_ize(self):
        """
        Normalize lambdas and make orthogonal.
        """
        def get_normalization(v):
            return np.sqrt(np.dot(v, v))

        self.lambdas[0] /= get_normalization(self.lambdas[0])
        for i in range(1, self.K):
            for j in range(0, i):
                v = np.dot(self.lambdas[i], self.lambdas[j])
                self.lambdas[i] -=  v * self.lambdas[j]
                    
            self.lambdas[i] /= get_normalization(self.lambdas[i])


    def project(self):
        """
        Project the model onto the data.
        """
        self.projections = np.dot(self.latents.T, self.lambdas)
