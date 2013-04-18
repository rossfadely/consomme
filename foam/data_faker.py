
import numpy as np
import matplotlib.pyplot as pl


def create_spectra(Nspec,sed1,sed2):
    """
    Return linear combinations of two base spectra
    """
    frac = np.random.rand(Nspec)
    spec = sed1[None,:] * frac[:,None] + \
    sed2[None,:] * (1. - frac[:,None])
    return spec

def default_fake(N,noise_level=3e-4,rescale=0.25,seed=None):
    data = np.loadtxt('../seds/Ell1_A_0.sed')
    lam1 = data[:,0]
    sed1 = data[:,1]

    data = np.loadtxt('../seds/Ell7_A_0.sed')
    lam2 = data[:,0]
    sed2 = data[:,1]

    assert np.sum(lam1-lam2)==0.0, 'Same wavelength grid'

    if seed!=None:
        np.random.seed(seed)

    spectra = create_spectra(N,sed1,sed2)
    noise = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    jitter = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    factor   = noise_level / rescale
    spectra /= factor
    noise   /= factor
    jitter  /= factor
    return lam1,spectra,spectra+noise+jitter,rescale


if __name__=='__main__':
    N = 10
    l,s,nl = default_fake(N)
    for i in range(N):
        pl.plot(l,s[i,:])
    pl.xlim(3000,10000)
    pl.show()
