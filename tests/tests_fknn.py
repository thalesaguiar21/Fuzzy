import os
import unittest

import numpy as np

from .context import fuzzy
from fuzzy.cluster.fknn import FKNN

DIRNAME = os.path.dirname(__file__)
FILEPATH = os.path.join(DIRNAME, 'dataset_blob.txt')

class TestsFKNN(unittest.TestCase):

    def load_blob(self):
        self.dataset = np.loadtxt(FILEPATH)
        self.Xtrain = self.dataset[:350, :-1]
        self.Xtest = self.dataset[350:, :-1]
        self.Ytrain = self.dataset[:350, -1]
        self.Ytest = self.dataset[350:, -1]

        ntrainsamples= self.Ytrain.size
        self.Ytrain = self.Ytrain.reshape(ntrainsamples, 1)

    def test_fit(self):
        data = self.load_blob()
        model = FKNN(2, 2, 2)
        model.fit(self.Xtrain, self.Ytrain)


