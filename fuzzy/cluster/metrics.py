import numpy as np

def minkowski(v, s, p):
    """ Compute the minkowski distance between two points

    Args:
        v (ndarray, 1d): first point
        s (ndarray, 1d): first point
        p (int): the order
    """
    if p < 1: 
        raise ValueError('p must be greater than 1')
    return np.sum(np.abs(v - s) ** p, axis=-1) ** (1/p)

