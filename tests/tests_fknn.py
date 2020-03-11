import unittest

import numpy as np

from .context import fuzzy
from fuzzy.cluster.fknn import fknn

class TestsFKNN(unittest.TestCase):

    def test_run(self):
        X, Y = np.arange(10).reshape(5, 2), np.arange(5)
        ukn = np.array([1, 2])
        fknn(ukn, X, Y, 10, 2)

