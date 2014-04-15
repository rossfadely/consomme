import time
import numpy as np

from bailey_empca import empca
from hmf import HMF

# settings
np.random.seed(1234)
nobs = 100
nvar = 200
nvec = 3

# Generate data, taken from S. Bailey -
# https://github.com/sbailey/empca/blob/master/empca.py
data = np.zeros(shape=(nobs, nvar))
x = np.linspace(0, 2*np.pi, nvar)
for i in range(nobs):
    for k in range(nvec):
        c = np.random.normal()
        data[i] += 5.0 * nvec / (k + 1) ** 2 * c * np.sin(x * (k + 1))

# add a linear mean
m = 2.
mean = m * x
data += mean[None, :]

#- Add noise
fs = 5
sigma = np.ones(shape=data.shape)
for i in range(nobs/10):
    sigma[i] *= fs
    sigma[i, 0:nvar/4] *= fs
ivar = 1.0 / sigma**2    
noisy_data = data + np.random.normal(scale=sigma)

# calculate mean to subtract off for Bailey EMPCA
ivar_mean = np.sum(noisy_data * ivar, axis=0) / np.sum(ivar, axis=0)


# Bailey empca, this spits out a bunch of lines...
t = time.time()
print '\n\n\n\n Running Bailey EMPCA'
em = empca(noisy_data - ivar_mean, ivar, niter=50)
print 'Done in %0.4fs' % (time.time() - t)

# HMF
t = time.time()
print '\n\n\n\nRunning HMF\n'
h = HMF(noisy_data, nvec, ivar)
print 'Done in %0.4fs' % (time.time() - t)
