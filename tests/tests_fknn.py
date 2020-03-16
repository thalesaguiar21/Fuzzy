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

    def test_accuracy(self):
        data = self.load_blob()
        model = FKNN(2, 2, 2)
        model.fit(self.Xtrain, self.Ytrain)
        preds = model.predict(self.Xtest)
        acc = accuracy(self.Ytest, preds) * 100
        self.assertGreater(acc, 90.0)

    def test_fuzzpred_rowsum(self):
        data = self.load_blob()
        model = FKNN(2, 2, 2)
        model.fit(self.Xtrain, self.Ytrain)
        preds = model.predict_fuzz(self.Xtest)
        rowsums = np.sum(preds, axis=1)
        areclose = rowsums - 1.0 > -1e-4
        self.assertTrue(areclose.all())

    def test_fuzzpred_colsum(self):
        data = self.load_blob()
        model = FKNN(2, 2, 2)
        model.fit(self.Xtrain, self.Ytrain)
        preds = model.predict_fuzz(self.Xtest)
        colsums = np.sum(preds, axis=0)
        atleast_nsamples = colsums <= self.Xtest.shape[0]
        self.assertTrue(atleast_nsamples.all())


def accuracy(reals, predictions):
    n_hits = 0
    for real, pred in zip(reals, predictions):
        if int(real) == int(pred):
            n_hits += 1
    return n_hits/len(reals)



