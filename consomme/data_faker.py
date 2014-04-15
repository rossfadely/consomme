import numpy as np

def noiseless_fake(N, Neig, scale=64., window=21, seed=None):
    """
    Taking some eigenspectra from SDSS/Yip '04, constructing a toy.
    """
    # load mean spectrum
    d = np.loadtxt('../seds/galaxyKL_meanSpec_regrid.dat')
    lamb = d[:, 0]
    mean = d[:, 1]

    # load eigenspectra
    eigspec = np.random.randn(lamb.shape[0], lamb.shape[0])
    for i in range(Neig):
        d = np.loadtxt('../seds/galaxyKL_eigSpec_'+str(i + 1)+'.dat')
        if i == 0:
            eiglamb = d[:, 0]
            ind = np.where((eiglamb >= lamb.min()) & 
                           (eiglamb <= lamb.max()))[0]
        eigspec[i, :] = d[ind, 1] / np.sqrt(np.sum(d[ind, 1]**2.))

    # made up crap to set eigvals
    eval_list = np.array([0.9, 0.05, 0.03, 0.02]) * scale
    eigvals = np.zeros_like(lamb)       
    assert Neig <= eval_list.shape[0]
    eigvals[:Neig] = eval_list[:Neig]

    cov = np.dot(eigspec.T, np.dot(np.diag(eigvals), eigspec))

    # make the data
    data = np.random.multivariate_normal(mean, cov, N)

    # smooth to make it less noisy
    if window > 0:
        w = np.ones(window, 'd')
        for i in range(N):
            data[i, (window - 1) / 2:-(window - 1) / 2] = \
                np.convolve(w / w.sum(), data[i, :], mode='valid')
        mean[(window - 1) / 2:-(window - 1) / 2] = \
                np.convolve(w / w.sum(), mean, mode='valid')
        
    return lamb, data, mean

def add_noise(lam, spectra, n_range=(1, 2), uniform_across=True,
              uniform_within=True):


    # rough shape of typical SDSS gal flux/noise spec
    pt1 = np.array([4000,2.]) 
    pt2 = np.array([7000,23.])  
    pt3 = np.array([9000,7.5])

    if uniform_across:
        scales = np.ones(spectra.shape[0]) * n_range[0]
    else:
        scales = np.random.rand(spectra.shape[0]) * \
            (n_range[1] + n_range[0]) + n_range[0]
        
    # lines defining two halves of kink
    m1s = scales * (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    m2s = scales * (pt3[1]-pt2[1])/(pt3[0]-pt2[0])
    b1s = scales * pt2[1] - m1s * pt2[0]
    b2s = scales * pt3[1] - m2s * pt3[0]

    # flux / noise ratios
    ratios = np.zeros_like(spectra)
    ratios = m1s[:,None] * lam[None,:] + b1s[:,None]
    ind = np.where(lam < pt1[0])[0]
    tmp = ratios[:,ind[-1]+1]
    ratios[:,ind] = tmp[:,None]
    ind = np.where(lam > pt2[0])[0]
    tmp = lam[ind]
    ratios[:,ind] = m2s[:,None] * tmp[None,:] + b2s[:,None]

    if uniform_within:
        mr = np.mean(ratios,axis=1)
        ratios *= 0.0
        ratios += mr[:,None]

    noise = 1.0 / ratios
    nm = noise.min()
    noise /= nm
    spectra /= nm

    noisy_spectra = spectra + \
        np.random.randn(spectra.shape[0],spectra.shape[1]) * noise

    return noisy_spectra, spectra, noise, nm
