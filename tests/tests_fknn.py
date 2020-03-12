import unittest

import numpy as np

from .context import fuzzy
from fuzzy.cluster.fknn import FKNN

class TestsFKNN(unittest.TestCase):

    def test_run(self):
        X = np.arange(10).reshape(5, 2)
        Y = np.array([0, 0, 1, 1, 1]).reshape(5, 1)
        ukn = np.array([1, 2])
        model = FKNN(2, 2, 2)
        model.fit(X, Y)
        pred = model.predict(ukn)


