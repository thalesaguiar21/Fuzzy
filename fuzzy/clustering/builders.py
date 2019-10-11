from .models import FCM, FGMM


def fcm(nclusters, fuzzyness):
    validate_fcm_arguments(nclusters, fuzzyness)
    return FCM(nclusters, fuzzyness)


def validate_fcm_arguments(nclusters, fuzzyness):
    if nclusters < 2:
        raise ValueError('There must be at least two clusters')
    if fuzzyness <= 1:
        raise ValueError('Cluster fuzzyness must be greater than 1')


def fgmm(ncomponents, threshold):
    validate_fgmm_arguments(ncomponents, threshold)
    return FGMM(ncomponents, threshold)


def validate_fgmm_arguments(ncomponents, threshold):
    if ncomponents < 2:
        raise ValueError('There must be at least two clusters')
    if threshold < 0.001:
        raise ValueError('Threshold must be at least 0.001')

