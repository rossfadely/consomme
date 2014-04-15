import numpy as np

#from hfa_single_nll import *
#from hfa_multi import *
from bailey_empca import *
from pca import *
from hmf import HMF

Nrun = 1
for j in range(Nrun):

    np.random.seed(1234)
    nobs = 100
    nvar = 200
    nvec = 3
    data = np.zeros(shape=(nobs, nvar))

    # Generate data, taken from S. Bailey -
    # https://github.com/sbailey/empca/blob/master/empca.py
    x = np.linspace(0, 2*np.pi, nvar)
    for i in range(nobs):
        for k in range(nvec):
            c = np.random.normal()
            data[i] += 5.0 * nvec / (k + 1) ** 2 * c * np.sin(x * (k + 1))

    # add a mean
    m = 2.
    mean = m * x
    data += mean[None, :]

    if j == 0:
        np.savetxt('../data/simple_toy_true_mean.dat', mean)

    #- Add noise
    fs = 5
    sigma = np.ones(shape=data.shape)
    for i in range(nobs/10):
        sigma[i] *= fs
        sigma[i, 0:nvar/4] *= fs
    weights = 1.0 / sigma**2    
    noisy_data = data + np.random.normal(scale=sigma)

    # mean estimates
    cm = np.mean(data, axis=0)
    m1 = np.mean(noisy_data, axis=0)
    m2 = np.sum(noisy_data * weights, axis=0) / np.sum(weights, axis=0)
    #np.savetxt('../data/simple_toy_cleanmean_%0.0f_%d.dat' % (fs, j), m1)
    #np.savetxt('../data/simple_toy_str8mean_%0.0f_%d.dat' % (fs, j), m1)
    #np.savetxt('../data/simple_toy_ivarmean_%0.0f_%d.dat' % (fs, j), m2)

    # true eigvec
    tevc, tevl = svd_pca(data)
    #np.savetxt('../data/simple_toy_true_eigvec_%d.dat' % j, tevc.T)

    # Weighted EMPCA
    em = empca(noisy_data - m2, weights, niter=50)
    #np.savetxt('../data/simple_toy_wempca_eigval_%0.0f_%d.dat' % (fs, j), 
    #           em.eigvec.T)

    h = HMF(noisy_data, nvec, weights)

    # HFA
    #hfa = HFAModel(noisy_data, sigma ** 2., nvec, single_nll_and_gradients,
    #               jitter_model=None, threads=10)
    #hfa.initialize()
    #hfa.run_inference(factr=1e4)
    #np.savetxt('../data/simple_toy_hfa_mean_%0.0f_%d.dat' % (fs, j), hfa.mean)
    #np.savetxt('../data/simple_toy_hfa_eigval_%0.0f_%d.dat' % (fs, j), hfa.lam)
