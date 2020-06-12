import numpy as np

def minkowski(V, S, p):
    if p < 1: 
        raise ValueError('p must be greater than 1')
    V = _validate_vector(V)
    S = _validate_vector(S)
    abssums = np.sum(np.abs(V - S) ** p, axis=1)
    return abssums ** (1/p)


def _validate_vector(vec):
    newvec = np.array(vec)
    if len(newvec.shape) == 1:
        newvec = np.array([newvec])
    return newvec


