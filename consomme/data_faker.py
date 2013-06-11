
import numpy as np
import matplotlib.pyplot as pl


def default_fake(N,noise_level=0.05,Ngauss=3,seed=None,
                wavelimits=(None,None),subtract_mean=False):
    """
    Taking some eigenspectra from SDSS/Yip '04, constructing a toy.
    Returned spectra are zero mean.
    """
    Neig = 2
    # load eigenspectra
    v = [1,4]
    for i in range(Neig):
        d = np.loadtxt('../seds/galaxyKL_eigSpec_'+str(v[i])+'.dat')
        if i==0:
            eigspec = np.zeros((Neig,d.shape[0]))
            eiglamb = d[:,0]
        eigspec[i,:] = d[:,1] / np.sqrt(np.sum(d[:,1]**2.))

    aas = np.zeros((N,2))
    a1s = np.random.rand(N) 
    a2s = a1s * np.random.rand(N) 
    aas[:,0] = a1s
    aas[:,1] = a2s
    aas /= np.sqrt(np.sum(aas**2.,axis=1))[:,None]

    # make the data
    data = np.sum(aas[:,:,None] * eigspec[None,:,:],axis=1)

    # smooth to make it less noisy
    wl = 21
    w = np.ones(wl,'d')
    for i in range(N):
        data[i,(wl-1)/2:-(wl-1)/2] = np.convolve(w/w.sum(),data[i,:],mode='valid')

    # trim wavelength range
    if wavelimits[0]!=None:
        ind  = np.where((eiglamb>wavelimits[0]) & (eiglamb<wavelimits[1]))[0]
        data = data[:,ind] 
        eiglamb = eiglamb[ind]
        
    # normalize
    ind = np.where((eiglamb > 4000) & (eiglamb < 5000))[0]
    spectra = data / np.mean(data[:,ind],axis=1)[:,None]

    # subtract mean
    if subtract_mean:
        spectra -= np.mean(spectra,axis=0)[None,:]

    # adjust noise
    noise_level *= np.mean(spectra)

    # noise, jitter
    noise = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    jitter = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    
    return eiglamb,spectra,spectra+noise+jitter,noise_level

def richer_fake(N,Neig,noise_level=0.05,window=21,seed=None,
                wavelimits=(None,None),subtract_mean=False):
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
    aas = np.zeros((N,Neig))
    assert Neig>1
    aas[:,0] = a1s
    aas[:,1] = a2s
    if Neig>2:
        aas[:,2] = a3s
        mag = np.max(np.sqrt(aas[:,2]**2.)) * 0.1
        for i in range(Neig-3):
            aas[:,i+3] = (np.random.rand(N) - 0.5) * 2 * mag
        mag *= 0.1
    aas /= np.sqrt(np.sum(aas**2.,axis=1))[:,None]

    # make the data
    data = np.sum(aas[:,:,None] * eigspec[None,:,:],axis=1)

    # smooth to make it less noisy
    w = np.ones(window,'d')
    for i in range(N):
        data[i,(window-1)/2:-(window-1)/2] = np.convolve(w/w.sum(),data[i,:],mode='valid')

    # trim wavelength range
    if wavelimits[0]!=None:
        ind  = np.where((eiglamb>wavelimits[0]) & (eiglamb<wavelimits[1]))[0]
        data = data[:,ind] 
        eiglamb = eiglamb[ind]
        
    # normalize
    ind = np.where((eiglamb > 4000) & (eiglamb < 5000))[0]
    spectra = data / np.mean(data[:,ind],axis=1)[:,None]

    # subtract mean
    if subtract_mean:
        spectra -= np.mean(spectra,axis=0)[None,:]

    # adjust noise
    noise_level *= np.mean(spectra)

    # noise, jitter
    noise = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    jitter = noise_level * np.random.randn(spectra.shape[0],
                                          spectra.shape[1])
    
    return eiglamb,spectra,spectra+noise+jitter,noise_level
