import numpy as np


def genbell(x, a, b, c):
    """ A generalised fuzzy bell function

    Args:
        x: the input
        a: function width
        b: plato spam
        c: center
    """
    if 0 in a:
        raise ValueError("Function spam must be greater than 0")
    denom =  np.abs((x-c)/a) ** (2*b)
    return 1 / (1+denom)


def neighbours(nneighbours, label_count, curr):
    mdegrees = (0.49/nneighbours) * label_count
    mdegrees[curr] += 0.51
    return mdegrees

