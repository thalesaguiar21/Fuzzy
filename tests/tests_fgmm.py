from .context import fuzzy
from fuzzy.models.clustering import FGMM
import numpy as np
import unittest


class TestsFGMM(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestsFGMM, self).__init__(*args, **kwargs)
        self.fgmm = FGMM(4)
        data = np.loadtxt('/home/thales/DEV/Fuzzy/tests/dataset_blob.txt')
        self.Xtrain = data[:350, :-1]
        self.Xtest = data[350:, :-1]
        self.fgmm.fit(self.Xtrain)

    def test_initialisation(self):
        pass

