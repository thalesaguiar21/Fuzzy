import unittest

import numpy as np

from .context import fuzzy
from fuzzy.cluster.fknn import FKNN
from . import Xtrain, Xtest, Ytrain, Ytest


_MODEL = FKNN(2, 2, 2)
_MODEL.fit(Xtrain, Ytrain)


class TestsPredict(unittest.TestCase):

    def setUp(self):
        self.model = _MODEL

    def test_prediction(self):
        preds = self.model.predict(Xtest)
        acc = accuracy(Ytest, preds) * 100
        self.assertGreater(acc, 90.0)

    def test_fuzzpred_rowsum(self):
        preds = self.model.predict_fuzz(Xtest)
        rowsums = np.sum(preds, axis=1)
        areclose = rowsums - 1.0 > -1e-4
        self.assertTrue(areclose.all())

    def test_fuzzpred_colsum(self):
        preds = self.model.predict_fuzz(Xtest)
        colsums = np.sum(preds, axis=0)
        atleast_nsamples = colsums <= Xtest.shape[0]
        self.assertTrue(atleast_nsamples.all())

    def test_no_fit_call(self):
        self.model = FKNN(2, 2, 2)
        self.assertRaises(ValueError, self.model.predict, Xtest)


def accuracy(reals, predictions):
    n_hits = 0
    for real, pred in zip(reals, predictions):
        if int(real) == int(pred):
            n_hits += 1
    return n_hits/len(reals)



