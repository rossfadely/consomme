import numpy as np


def svd_pca(data):
    """
    Run PCA using a singular value decomposition.
    
    `data` is a numpy array, n_samples by n_dimensions
    """
    U, S, V = np.linalg.svd(data)
    eigvec = V.T
    eigval = S ** 2. / (data.shape[0] - 1.)
    return eigvec, eigval
