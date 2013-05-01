
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

def default_fake(N,noise_level=0.05,seed=None):
    data = np.loadtxt('../seds/Ell1_A_0.sed')
    lam1 = data[:,0]
    sed1 = data[:,1]

    data = np.loadtxt('../seds/Ell7_A_0.sed')
    lam2 = data[:,0]
    sed2 = data[:,1]

    assert np.sum(lam1-lam2)==0.0, 'Same wavelength grid'

    if seed!=None:
        np.random.seed(seed)

    spectra = create_spectra(N,sed1,sed2) * 500.
    noise_level *= np.mean(spectra)
    noise = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    jitter = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    return lam1,spectra,spectra+noise+jitter,noise_level

def richer_fake(N,Neig,noise_level=0.05,Ngauss=3,seed=None,
                wavelimits=(None,None)):
    """
    Taking some eigenspectra from SDSS/Yip '04, constructing a toy.
    Returned spectra are zero mean.
    """

    # load eigenspectra
    for i in range(Neig):
        d = np.loadtxt('../seds/galaxyKL_eigSpec_'+str(i+1)+'.dat')
        if i==0:
            eigspec = np.zeros((Neig,d.shape[0]))
            eiglamb = d[:,0]
        eigspec[i,:] = d[:,1] / np.sqrt(np.sum(d[:,1]**2.))

    # coefficients to give something like early types
    a1s = np.random.rand(N) * 0.1 + 0.9
    phi = np.random.rand(N) * 12.5 + 7.5
    a2s = a1s * np.tan(phi*np.pi/180.)
    tta = np.random.rand(N) * 6 + 86
    a3s = np.cos(tta*np.pi/180.)
    aas = np.zeros((N,3))
    aas[:,0] = a1s
    aas[:,1] = a2s
    aas[:,2] = a3s
    aas /= np.sum(aas,axis=1)[:,None]

    # make the data
    data = np.sum(aas[:,:,None] * eigspec[None,:,:],axis=1)

    # smooth to make it less noisy
    wl = 21
    w = np.ones(wl,'d')
    for i in range(N):
        data[i,(wl-1)/2:-(wl-1)/2] = np.convolve(w/w.sum(),data[i,:],mode='valid')

    # trim wavelength range
    if wavelimits[0]!=None:
        ind  = np.where((eiglamb>wavelimits[0]) & (eiglamb<wavelimits[1]))
        data = data [:,ind] 
        
    # multiply to get mean to order unity
    spectra = data / np.mean(data)

    # adjust noise
    noise_level *= np.mean(spectra)

    # noise, jitter
    noise = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    jitter = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    
    return eiglamb,spectra,spectra+noise+jitter,noise_level
