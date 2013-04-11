import numpy as np
import matplotlib.pyplot as pl

from empca import *

class Bubble(object):
    """
    Factor Analysis deconvolution, using online learning.
    """
    def __init__(self,data,obsvar,latent_dim,Kmax):
        """

        """
        assert data.shape==obsvar.shape, 'Data and obs. ' + \
            'variance have different shapes'

        self.N = data.shape[0]
        self.D = data.shape[1]
        self.Kmax = Kmax
        self.K = 1
        self.M = latent_dim
        self.jitters = np.zeros(self.D)

        # constant learning rates... fix soon.
        self.mean_rate = 1.e-8 # ok for now...
        self.jitter_rate = 0.1
        self.lambda_rate = 0.1

        # Suffling the data
        ind = np.random.permutation(self.N)
        self.data = data[ind,:]
        self.obsvar = obsvar[ind,:]

        self.initialize()
        self.initial_mean = self.mean.copy()
        self.run_inference()

    def initialize(self):

        # inverse variance weighted mean
        self.mean = np.sum(self.data / self.obsvar,axis=0) / \
            np.sum(1.0 / self.obsvar,axis=0)

        # use EM PCA to initialize factor loadings
        if self.M==0:
            self.lam = np.zeros()
        else:
            empca = EMPCA(self.data.T,self.M)
            self.lam = empca.lam.T
            
    def invert_cov(self):
        """
        Make inverse covariance, using the inversion lemma.
        """
        if self.M==0:
            self.inv_cov = np.diag(1.0 / (self.jitters+self.variance))
        else:
            # repeated in lemma
            lam = self.lam
            lamT = lam.T
            psiI = np.diag(1.0 / (self.jitters + self.variance))
            psiIlamT = np.dot(psiI,lamT)

            # the lemma
            bar = np.dot(lam,psiI)
            foo = np.linalg.inv(np.eye(self.M) + np.dot(lam,psiIlamT))
            self.inv_cov = psiI - np.dot(psiIlamT,np.dot(foo,bar))

    def run_inference(self):

        for ii in range(100):

            i = np.mod(ii,self.data.shape[0])
            
            self.datum = self.data[i,:]
            self.variance = self.obsvar[i,:]

            self.invert_cov()
            self.make_gradient_step()

    def make_gradient_step(self):
        for d in range(self.D):
            self.mean[d] -= self.mean_rate * self.mean_gradient(d)
            print self.jitter_gradient(d)
            #self.jitter[d] -= self.mean_rate * self.mean_gradient(d)

    def mean_gradient(self,j):
        return -2. * np.dot((self.datum - self.mean),self.inv_cov[:,j])
        
    def jitter_gradient(self,j):
        pt1 = np.dot(self.inv_cov[j,:],(self.datum - self.mean).T)
        pt2 = np.dot((self.datum - self.mean),self.inv_cov[:,j])
        return self.D * self.inv_cov[j,j] - np.dot(pt1,pt2)

        
