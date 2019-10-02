import numpy as np
from numpy.linalg import inv as minverse
from abc import ABC, abstractmethod


class _LSE(ABC):
    """ Interface for Least Square Estimation methods """
    @abstractmethod
    def solve(self, coeficients, results):
        ...


class Recursive(_LSE):
    """ Recursive LSE using a forgeting factor and confidence score """
    _MIN_CONFIDENCE = 1e-5
    _MAX_CONFIDENCE = 1000
    _MIN_FORGETRATE = 1e-4
    _MAX_FORGETRATE = 1.0

    def __init__(self, forgetrate, confidence):
        """
        Parameters
        ----------
        forgetrate : double,
            The forgetting factor between 1e-5 and 1.0
        confidence : double,
            The initial confidence between 1 and 1000
        """
        self.forgetrate = clip(
            forgetrate, self._MIN_FORGETRATE, self._MAX_FORGETRATE)
        self.confidence = clip(
            confidence, self._MIN_CONFIDENCE, self._MAX_CONFIDENCE)

    def solve(self, coeficients, results):
        """ Approximate the parameters for the given system, such that
        AX = B + e.
        """
        qtd_variables = coeficients.shape[1]
        qtd_equations = coeficients.shape[0]
        covariances = np.eye(qtd_variables) * self.confidence
        theta = np.zeros(qtd_variables)
        for k in range(qtd_equations):
            coefs_k = np.array([coeficients[k, :]]).T
            term = covariances @ coefs_k
            denom = coefs_k.T @ term + self.forgetrate
            gain = (term / denom).reshape(qtd_variables, )
            theta += gain*(results[k] - coefs_k.T@theta)
            part = covariances @ coefs_k @ coefs_k.T @ covariances
            covariances -= part / denom
            covariances *= 1.0 / self.forgetrate
        return theta


def clip(value, lower, upper):
    """ clip a value between lower and upper """
    if upper < lower:
        lower, upper = upper, lower  # Swap variables
    return min(max(value, lower), upper)


class Matricial(_LSE):
    """ A simple matricial solver for AX = B linear systems """

    def solve(self, coeficients, results):
        """ Approximate parameters X, such that AX = B + e """
        coeficients_pinverse = minverse(coeficients.T @ coeficients)
        prediction = coeficients_pinverse @ coeficients.T @ results
        return prediction
