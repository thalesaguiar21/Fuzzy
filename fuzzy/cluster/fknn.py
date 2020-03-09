import numpy as np


def fknn(ukn, X, Y, k, p):
    n_points, dim = X.shape
    k = clip(k, 1, n_points)
    if Y.size != n_points:
        raise ValueError('Different number of labels and samples')

    neighbours = [Y[0]]
    dists = [minkowski(ukn, X[0], p)]
    for i in range(1, k):
        dist = minkowski(ukn, X[i], p)
        j = 0
        while j < i and dist > dists[j]:
            j += 1
        dists.insert(j, dist)
        neighbours.insert(j, Y[i])

    i = k + 1
    while i < n_points:
        dist = minkowski(ukn, X[i])
        if i < k:
            dists.append(dist)
        else:
            for j in range(len(dists)):
                if dists[j] > dist:
                    dists.append(dist)
                    dists.pop(j)
                    neighbours.pop(j)
                    neighbours.insert(j, Y[i])
        i += 1


def clip(x, lower, higher):
    return min(max(x, lower), higher)


def minkowski(a, b, p):
    ''' Computes the Minkowki distance between two points
    '''
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        raise ValueError('Cannot compute with different array dimensions')
    pow_dists = np.abs(a - b)**p
    return np.abs(pow_dists).sum() ** (1.0/p)


X, Y = np.arange(10).reshape(5, 2), np.arange(5)
ukn = np.array([1, 2])
fknn(ukn, X, Y, 10, 2)

