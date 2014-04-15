import numpy as np

from sklearn.decomposition import FactorAnalysis
from scipy.optimize import fmin_l_bfgs_b


class HFAModel(object):
    """
    Heteroskedastic Factor Analysis, using batch optimization.

    See https://github.com/rossfadely/consomme for model details.

    calling arguments:
    ------------------

    `data`:  numpy Nd array with shape N samples by D features.
             This is the data to factorize.
    `obsvar`: numpy Nd array with shape == `data.shape`.  This
              array corresponds to the observed variances - if
              this is the same for each datum, you should be
              doing vanilla FA!
    `latent_dim`: Integer dimensionality of the low-rank part of the
                  covariance matrices.
    `max_fun`: Integer number of calls to the neg. log likelihood
              function for fmin_l_bfgs_b. See http://bit.ly/18bCypv
    `factr`: Tolerance factor for optimization convergence.
             See http://bit.ly/18bCypv
    `jitter_model`: Either 'full', 'one', or None. 'full' allows a
                    jitter term for each element of the diagonal,
                    'one' forces one value for all the diagonal
                    elements, and None uses only the observed
                    variances.

    TODO -
    ------
    - Check all jitter modes.
    - Double check jitter gradients.

    """
    def __init__(self, data, obsvar, latent_dim, max_fun=1e6, factr=1e7,
                 jitter_model='one'):

        assert data.shape == obsvar.shape, 'Data and obs. ' + \
            'variance have different shapes'

        # jitter options
        jtypes = ['full', 'one', None]
        assert jitter_model in jtypes, 'Jitter model must be \'full\', ' + \
            '\one\', or None'

        # assign
        self.M = latent_dim
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.data = data
        self.eyeM = np.eye(self.M)
        self.psiI = np.zeros((self.D, self.D))
        self.jtype = jitter_model
        self.obsvar = obsvar
        self.psi_diag_inds = np.diag_indices_from(self.psiI)

        #self.initialize()
        #self.run_inference(max_fun, factr)

    def initialize(self):
        """
        Initialize the model.
        """
        # inverse variance weighted mean
        if np.sum(self.obsvar) != 0.0:
            self.mean = np.sum(self.data / self.obsvar, axis=0) / \
                np.sum(1.0 / self.obsvar, axis=0)
        else:
            self.mean = np.mean(self.data, axis=0)

        # use Factor Analysis to initialize factor loadings
        if self.M == 0:
            self.lam = np.zeros(1)
        else:
            fa = FactorAnalysis(n_components=self.M)
            fa.fit(self.data)
            self.lam = fa.components_.T

        # initialize jitter
        if self.jtype is None:
            self.jitter = np.array([])
        elif self.jtype is 'one':
            self.jitter = 0.0
        else:
            self.jitter = np.zeros(self.D)

        # save a copy
        self.initial_mean = self.mean.copy()
        self.initial_jitter = self.jitter.copy()
        self.initial_lambda = self.lam.copy()

    def run_inference(self, max_fun=1e6, factr=1e7):
        """
        Optimize the parameters using fmin_l_bfgs_b
        """
        # initial parms
        p0 = self.make_parm_list(self.mean, self.lam.ravel(), self.jitter)

        # bounds
        b = self.make_bounds()

        result = fmin_l_bfgs_b(self.call_nll, p0, self.calc_gradients,
                               bounds=b, factr=factr, maxfun=max_fun,
                               iprint=2)
        self.assign_parms(result[0])

    def call_nll(self, p):
        """
        Make a negative log likelihood call for the model.
        """
        self.assign_parms(p)
        self.grads = np.zeros_like(p)
        nll = self.total_negative_log_likelihood()
        return nll

    def calc_gradients(self, p):
        """
        Calculate gradients for fmin_l_bfgs_b
        """
        return self.grads

    def assign_parms(self, p):
        """
        Assign the parameter from the list given by fmin_l_bfgs_b
        """
        self.mean = p[:self.D]
        self.lam = p[self.D:self.D * (self.M + 1)].reshape(self.lam.shape)
        if self.jtype is not None:
            self.jitter = p[self.D * (self.M + 1):]

    def make_parm_list(self, m, l, j):
        """
        Make the initial parm list for fmin_l_bfgs_b
        """
        size = m.shape[0] + l.shape[0]
        if self.jtype == 'one':
            size += 1
        if self.jtype == 'full':
            size += j.shape[0]
        p0 = np.empty(size)
        p0[:self.D] = m
        p0[self.D:self.D * (self.M + 1)] = l
        if self.jtype is not None:
            assert 0
            p0[self.D * (self.M + 1):] = j
        return p0

    def make_bounds(self):
        """
        Make parameter bounds for fmin_l_bfgs_b
        """
        # mean bounds
        m = [(-1. * np.Inf, np.Inf) for i in range(self.D)]
        # lambda bounds
        l = [(-1. * np.Inf, np.Inf) for i in range(self.D * self.M)]

        bounds = m
        bounds.extend(l)
        if self.jtype is not None:
            # jitter bounds
            j = [(0.0, np.Inf) for i in range(self.jitter.shape[0])]
            bounds.extend(j)

        return bounds

    def do_precalcs(self):
        """
        Calculate matrices used repeatedly below
        """
        self.psid = self.variance
        if self.jtype is not None:
            self.psid += self.jitter
        self.invert_cov()
        self.dmm = self.datum - self.mean
        self.dmmi = np.dot(self.dmm, self.inv_cov)

    def invert_cov(self):
        """
        Make inverse covariance, using the inversion lemma.
        """
        if self.M == 0:
            self.inv_cov = np.diag(1.0 / self.psid)
        else:
            if np.sum(self.psid) != 0.0:
                psiI = np.diag(1.0 / self.psid)
                psiIlam = np.zeros((self.D, self.M))
                # double check this is actually faster, generally.
                psiIlam = self.crazy_dot(psiI, self.lam, psiIlam)

                # the lemma, last step is slowest
                foo = np.linalg.inv(self.eyeM + np.dot(self.lam.T, psiIlam))
                self.inv_cov = psiI
                self.inv_cov -= np.dot(psiIlam, np.dot(foo, psiIlam.T))
            else:
                # fix to lemma-like version
                self.inv_cov = np.linalg.inv(np.dot(self.lam, self.lam.T))



    def total_negative_log_likelihood(self):
        """
        Return total negative log likelihood of current model
        """
        totnll = 0.0
        for i in range(self.N):
            self.datum = self.data[i, :]
            self.variance = self.obsvar[i, :]
            self.do_precalcs()
            totnll += self.single_negative_log_likelihood()
            self.grads[:self.D] += self.mean_gradients()
            self.grads[self.D:self.D * (self.M + 1)] += \
                self.lambda_gradients().ravel()
            if self.jtype is not None:
                self.grads[self.D * (self.M + 1):] += self.jitter_gradients()

        self.grads /= self.N

        return totnll

    def single_negative_log_likelihood(self):
        """
        Return the NLL of a single sample.
        """
        sgn, logdet = self.single_slogdet_cov()
        assert sgn > 0
        pt2 = np.dot(self.dmm, np.dot(self.inv_cov, self.dmm.T))
        return logdet + pt2

    def single_slogdet_cov(self):
        """
        Return sign and value of log det of covariance.
        """
        lamlamT = np.dot(self.lam,self.lam.T)
        return np.linalg.slogdet(lamlamT + np.diag(self.psid))

    def _check_one_gradient(self, kind, ind, eps=1.0e-6):
        """
        Gradient checking foo.
        """
        if kind == 'mean':
            parm = self.mean
        elif kind == 'jitter':
            parm = self.jitter
        else:
            parm = self.lam

        h = np.abs(parm[ind]) * eps
        if h == 0.0:
            h = eps

        parm[ind] += h
        self.do_precalcs()
        l1 = self.single_negative_log_likelihood()
        parm[ind] -= h
        self.do_precalcs()
        l2 = self.single_negative_log_likelihood()

        return (l1 - l2) / (h)
