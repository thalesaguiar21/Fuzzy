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
    plato = np.ones(means.shape)
    mfparams = np.array([plato, mfspam, means])
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

