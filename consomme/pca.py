import numpy as np


def svd_pca(data):
    """
    Run PCA using a singular value decomposition.
    
    `data` is a numpy array, n_samples by n_dimensions
    """
    U, S, eigvec = np.linalg.svd(data)
    eigval = S ** 2. / (data.shape[0] - 1.)
    return eigvec, eigval
