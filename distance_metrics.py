import numpy as np


def mahalanobis(p, m, cov_inv):
    '''Computes the Mahalanobis distance between two points (1-D arrays)
    
    Parameters
    ----------
    p : array_like
        First point
    m : array_like
        Second point
    cov_inv : array_like
        Inverse of the covariance matrix

    Returns
    -------
    float : Mahalanobis distance
    '''
    delta = p - m
    mahal = np.dot(np.dot(delta, cov_inv), delta)
    return np.sqrt(mahal)
