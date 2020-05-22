import numpy as np


def build_genmf(inputs, nmfs=2):
    """ Builds a fuzzy set of generalised bell functions
    
    Args:
        inputs: the input space
        nmfs: number of membership functions per input
    """
    feat_spams, mins = _find_feat_spam(inputs)
    means = _find_means(feat_spams, mins, nmfs)
    mfspam = np.repeat(feat_spams, nmfs)
    if np.any(mfspam == 0):
        raise ValueError("Can't create a feature MFs for feature space of 0"
                         "width! Please, check your data")
    plato = np.ones(means.shape)
    mfparams = np.array([mfspam, plato, means])
    return mfparams.T


def _find_feat_spam(inputs):
    mins = np.min(inputs, axis=0)
    maxs = np.max(inputs, axis=0)
    inp_spaces = np.abs(maxs - mins)
    return inp_spaces, mins


def _find_means(featspam, mins, nmfs):
    stepsizes = featspam / (nmfs-1)
    stepsizesrow = stepsizes.reshape(1, len(stepsizes))
    steps = np.arange(nmfs).reshape(nmfs, 1)
    splits = steps @ stepsizesrow + mins
    return splits.T.reshape(-1)

