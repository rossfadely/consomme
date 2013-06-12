import numpy as np
import matplotlib.pyplot as pl
from time import time

from empca import *
from rotate_factor import ortho_rotation

class FAModel(object):
    """
    Factor Analysis deconvolution, using online learning.
    """
    def __init__(self,data,obsvar,latent_dim,
                 max_iter=100000,compute_total_nll=False,
                 learning_init_relsizes=[0.01,0.01,0.01],
                 Ninit_est=10,check_rate=100,Neff=10,
                 decay_t0_factor=1.,decay_pow=0.5):
        """

        """
        assert data.shape==obsvar.shape, 'Data and obs. ' + \
            'variance have different shapes'

        self.N = data.shape[0]
        self.D = data.shape[1]
        self.M = latent_dim
        self.psiI = np.zeros((self.D,self.D))
        self.eyeM = np.eye(self.M)
        self.avg_factor = np.exp(-1./Neff)
        self.jitter = np.zeros(self.D)
        self.decay_t0 = decay_t0_factor * self.N
        self.decay_pow = decay_pow
        self.running_nll = np.ones(max_iter/check_rate) * -np.Inf
        self.psi_diag_inds = np.diag_indices_from(self.psiI)

        # Suffling the data
        ind = np.random.permutation(self.N)
        self.data = data[ind,:]
        self.obsvar = obsvar[ind,:]

        self.initialize()
        self.initial_mean   = self.mean.copy()
        self.initial_jitter = self.jitter.copy()
        self.initial_lambda = self.lam.copy()
        if compute_total_nll:
            self.initial_nll = self.total_negative_log_likelihood()
        self.calc_init_rates(Ninit_est,learning_init_relsizes)
        self.run_inference(max_iter,check_rate)

    def initialize(self):

        # inverse variance weighted mean
        if np.sum(self.obsvar)!=0.0:
            self.mean = np.sum(self.data / self.obsvar,axis=0) / \
                np.sum(1.0 / self.obsvar,axis=0)
        else:
            self.mean = np.mean(self.data,axis=0)

        # use EM PCA to initialize factor loadings
        if self.M==0:
            self.lam = np.zeros(1)
        else:
            empca = EMPCA(self.data.T,self.M)
            R = ortho_rotation(empca.lam,method='varimax')
            self.lam = np.dot(empca.lam,R)

    def run_inference(self,max_iter,check_rate):

        j = 0
        for ii in range(max_iter):

            # shuffle
            if (ii%self.N)==0:
                ind = np.random.permutation(self.N)
                self.data = self.data[ind,:]
                self.obsvar = self.obsvar[ind,:]

            # specify datum
            i = np.mod(ii,self.data.shape[0])            
            self.datum = self.data[i,:]
            self.variance = self.obsvar[i,:]
            t = time()
            self.do_precalcs()
            print time()-t

            # Neg. log likelihood
            if (ii%check_rate)==0:
                j = self.estimate_nll(j)
                if (j%10)==0: print ii,j,self.running_nll[j]
                # convergence test...

            # not converged, make a step
            t = time()
            self.calc_rates(j)
            print time()-t
            t = time()
            self.make_gradient_step()
            print time()-t
            print ii
            if ii==1:
                assert 0
     
    def estimate_nll(self,j):
        est_nll = self.single_negative_log_likelihood()
        if j==0:
            self.running_nll[j] = est_nll
        else:
            self.running_nll[j] = self.running_nll[j-1] * self.avg_factor + \
                (1. - self.avg_factor) * est_nll
        return j+1

    def do_precalcs(self):
        """
        Calculate matrices used repeatedly below
        """
        self.psid = self.jitter + self.variance
        self.invert_cov()
        self.dmm = self.datum - self.mean
        self.dmmi = np.dot(self.dmm,self.inv_cov)
     
    def invert_cov(self):
        """
        Make inverse covariance, using the inversion lemma.
        """
        if self.M==0:
            self.inv_cov = np.diag(1.0 / self.psid)
        else:
            if np.sum(self.psid)!=0.0:
                psiI = np.diag(1.0 / self.psid)
                psiIlam = np.zeros((self.D,self.M))
                psiIlam = self.crazy_dot(psiI,self.lam,psiIlam)

                # the lemma, last step is slowest
                foo = np.linalg.inv(self.eyeM + np.dot(self.lam.T,psiIlam))
                self.inv_cov = psiI
                self.inv_cov -= np.dot(psiIlam,np.dot(foo,psiIlam.T))
            else:
                # fix to lemma-like version
                self.inv_cov = np.linalg.inv(np.dot(self.lam,self.lam.T))

    def crazy_dot(self,a,b,result):
        """
        Somehow this is faster...
        """
        for m in range(self.M):
            result[:,m] = np.dot(a,b[:,m])
        return result
            
    def make_gradient_step(self):

        self.mean -= self.mean_rate * self.mean_gradients()
        self.jitter -= self.jitter_rate * self.jitter_gradients()
        self.lam -= self.lambda_rate * self.lambda_gradients()

        ind = (self.jitter < 0.0)
        self.jitter[ind] = 0.0

    def calc_rates(self,t):

        t0 = self.decay_t0 * self.N
        p  = self.decay_pow

        self.mean_rate *= (t0 / (t0+t)) ** p
        self.jitter_rate *= (t0 / (t0+t)) ** p
        self.lambda_rate *= (t0 / (t0+t)) ** p

    def calc_init_rates(self,Ninit_est,learning_init_relsizes):

        self.mean_rate = np.Inf
        self.jitter_rate = np.Inf
        self.lambda_rate = np.Inf

        mf = learning_init_relsizes[0]
        jf = learning_init_relsizes[1]
        lf = learning_init_relsizes[2]

        ind = np.random.permutation(self.N)
        for i in range(Ninit_est):
            self.datum = self.data[ind[i],:]
            self.variance = self.obsvar[ind[i],:]
            self.do_precalcs()

            self.mean_rate = np.minimum(self.mean_rate,
                                        np.min(mf/np.abs(self.mean_gradients())))
            self.jitter_rate = np.minimum(self.jitter_rate,
                                        np.min(jf/np.abs(self.jitter_gradients())))
            self.lambda_rate = np.minimum(self.mean_rate,
                                        np.min(lf/np.abs(self.lambda_gradients())))
        
    def mean_gradients(self):
        return -2. * self.dmmi
        
    def jitter_gradients(self):
        return self.D * self.inv_cov[self.psi_diag_inds] - self.dmmi * self.dmmi

    def lambda_gradients(self):
        pt1 = np.dot(self.inv_cov,self.lam) 
        pt2 = np.dot(self.dmm,pt1)
        v = self.dmm[:,None] * pt2[None,:]
        pt2 = np.zeros((self.D,self.M)) # crazy, this is faster.
        for m in range(self.M):
            pt2[:,m] = np.dot(self.inv_cov,v[:,m])
        return 2. * (self.D * pt1 - pt2)
    
    def total_negative_log_likelihood(self):
        totnll = 0.0
        for i in range(self.N):
            self.datum = self.data[i,:]
            self.variance = self.obsvar[i,:]
            self.do_precalcs()
            totnll += self.single_negative_log_likelihood()
        return totnll
        
    def single_negative_log_likelihood(self):
        sgn, logdet = self.single_slogdet_cov()
        assert sgn>0
        pt1 = self.D * logdet
        pt2 = np.dot(self.dmm,np.dot(self.inv_cov,self.dmm.T))
        return (pt1 + pt2)

    def make_cov(self):
        psi = np.diag(self.psid)
        lamlamT = np.dot(self.lam,self.lam.T)
        return psi+lamlamT

    def single_slogdet_cov(self):
        """
        Return sign and value of log det of covariance,
        using the matrix determinant lemma
        """
        pt1 = self.eyeM + np.dot(self.lam.T,self.lam / self.psid[:,None])
        det = np.linalg.det(pt1) * np.prod(self.psid)
        sgn = 1
        if det<0:
            sgn = -1
        return sgn, np.log(np.abs(det))

    def _check_gradients(self,Ncheck = 10):

        mgrad = self.mean_gradients()
        jgrad = self.jitter_gradients()
        lgrad = self.lambda_gradients()
        Dsamp = np.random.permutation(self.D)
        Msamp = np.random.permutation(self.M)
        
        for ii in range(Ncheck):

            i = np.mod(ii,self.D) 
            D = Dsamp[i]
            i = np.mod(ii,self.M) 
            M = Msamp[i]
            
            estm = self._check_one_gradient('mean',D)
            estj = self._check_one_gradient('jitter',D)
            estl = self._check_one_gradient('lambda',(D,M))
            
            print '\n\nCheck #%d, D = %d M = %d' % (ii,D,M)
            print '(Used mean grad., ' + \
                'Est. mean grad) = %g %g' % (mgrad[D],estm)
            print '(Used jitter grad., ' + \
                'Est. jitter grad) = %g %g' % (jgrad[D],estj)
            print '(Used lambda grad., ' + \
                'Est. lamda grad) = %g %g' % (lgrad[D,M],estl)

    def _check_one_gradient(self,kind,ind,eps = 1.0e-3):

        if kind=='mean':
            parm = self.mean
        elif kind=='jitter':
            parm = self.jitter
        else:
            parm = self.lam

        h = np.abs(parm[ind]) * eps
        if h==0.0: h = eps

        parm[ind] += h
        self.invert_cov()
        l1 = self.single_negative_log_likelihood()
        parm[ind] -= h
        self.invert_cov()
        l2 = self.single_negative_log_likelihood()

        return (l1 - l2) / (h)



"""
in lambda gradients -
        pt2 = np.zeros(self.M)
        for m in range(self.M):
            tmp = np.dot(self.inv_cov,self.lam[:,m])
            pt2[m] = np.dot(self.dmm,tmp) # scalar

        pt2 = np.dot(self.inv_cov,v)

"""
