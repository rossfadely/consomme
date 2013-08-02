import numpy as np

def single_nll_and_gradients(*args):
    args = args[0]
    datum = args[0]
    variance = args[1]
    mean = args[2]
    lam = args[3]
    lamT = args[4]
    jtype = args[5]
    Ncomponents = args[6]

    def invert_cov(variance, lam, lamT, Ncomponents):
        """
        Make inverse covariance, using the inversion lemma.
        """
        psiI = np.diag(1.0 / variance)
        if Ncomponents == 0:
            inv_cov = psiI
        else:
            if np.sum(variance) != 0.0:
                psiIlam = np.dot(psiI, lam)

                # the lemma, last step is slowest
                foo = np.linalg.inv(np.eye(Ncomponents) +
                                    np.dot(lamT, psiIlam))
                inv_cov = psiI
                inv_cov -= np.dot(psiIlam, np.dot(foo, psiIlam.T))
            else:
                # fix to lemma-like version
                inv_cov = np.linalg.inv(np.dot(lam, lamT))
        return inv_cov

    def do_precalcs(datum, variance, lam, lamT, Ncomponents):
        """
        Calculate matrices used repeatedly below
        """
        inv_cov = invert_cov(variance, lam, lamT, Ncomponents)
        dmm = datum - mean
        dmmi = np.dot(dmm, inv_cov)
        return inv_cov, dmm, dmmi

    def single_slogdet_cov(variance, lam, lamT):
        """
        Return sign and value of log det of covariance.
        """
        lamlamT = np.dot(lam, lamT)
        return np.linalg.slogdet(lamlamT + np.diag(variance))

    def mean_gradients(dmmi):
        """
        Return gradient of mean.
        """
        return -2. * dmmi

    def lambda_gradients(inv_cov, dmm, lam, Ncomponents):
        """
        Return gradients for factor loadings.
        """
        pt1 = np.dot(inv_cov, lam)
        pt2 = np.dot(dmm, pt1)
        v = dmm[:, None] * pt2[None, :]
        # crazy, this is faster.
        pt2 = np.zeros((inv_cov.shape[0], Ncomponents))
        for m in range(Ncomponents):
            pt2[:, m] = np.dot(inv_cov, v[:, m])
        return 2. * (pt1 - pt2)

    def jitter_gradients(inv_cov, dmmi, jtype):
        """
        Return jitter gradient - Ross double check.
        """
        jgrad = inv_cov[np.diag_indices_from(inv_cov)] - dmmi * dmmi
        if jtype is 'one':
            jgrad = np.mean(jgrad)
        return jgrad


    inv_cov, dmm, dmmi = do_precalcs(datum, variance, lam, lamT, Ncomponents)

    # negative log likelihood
    sgn, logdet = single_slogdet_cov(variance, lam, lamT)
    assert sgn > 0
    nll = logdet + np.dot(dmm, np.dot(inv_cov, dmm.T))

    # gradients
    mg = mean_gradients(dmmi)
    lg = lambda_gradients(inv_cov, dmm, lam, Ncomponents).ravel()
    if jtype is not None:
        jg = jitter_gradients(inv_cov, dmmi, jtype)
        return nll, mg, lg, jg
    else:
        return nll, mg, lg
