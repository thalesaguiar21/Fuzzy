import unittest

import numpy as np

from .context import fuzzy
from fuzzy.cluster.fknn import FKNN
from . import Xtrain, Xtest, Ytrain, Ytest


_MODEL = FKNN(2, 2, 2)
Ytrain_ = Ytrain.reshape(-1).astype(np.int32)
_MODEL.fit(Xtrain, Ytrain_)


class TestsPredict(unittest.TestCase):

    def setUp(self):
        self.model = _MODEL

    def test_prediction_with_complete_mdegs(self):
        preds = self.model.predict(Xtest)
        acc = accuracy(Ytest, preds) * 100
        self.assertGreater(acc, 90.0)

    def test_prediction_with_mean_mdegs(self):
        model = FKNN(2, 2, 2)
        model.fit(Xtrain, Ytrain_, init_strat='means')
        preds = model.predict(Xtest)
        acc = accuracy(Ytest, preds) * 100
        self.assertGreater(acc, 90.0)

    def test_prediction_with_knn_mdegs(self):
        model = FKNN(2, 2, 2)
        model.fit(Xtrain, Ytrain_, init_strat='knn')
        preds = model.predict(Xtest)
        acc = accuracy(Ytest, preds) * 100
        self.assertGreater(acc, 90.0)

    def test_fuzzpred_rowsum(self):
        preds = self.model.predict_fuzz(Xtest)
        rowsums = np.sum(preds, axis=1)
        areclose = rowsums - 1.0 > -1e-4
        self.assertTrue(areclose.all())

    def test_fuzzpred_rowsum_with_mean_mdges(self):
        model = FKNN(2, 2, 2)
        model.fit(Xtrain, Ytrain_, init_strat='means')
        preds = model.predict_fuzz(Xtest)
        rowsums = np.sum(preds, axis=1)
        areclose = abs(rowsums - 1) < 1e-4
        self.assertTrue(areclose.all())

    def test_fuzzpred_rowsum_with_knn_mdegs(self):
        model = FKNN(2, 2, 2)
        model.fit(Xtrain, Ytrain_, init_strat='knn')
        preds = model.predict_fuzz(Xtest)
        rowsums = np.sum(preds, axis=1)
        areclose = abs(rowsums - 1) < 1e-4
        self.assertTrue(areclose.all())

    def test_fuzzpred_colsum(self):
        preds = self.model.predict_fuzz(Xtest)
        colsums = np.sum(preds, axis=0)
        atleast_nsamples = colsums <= Xtest.shape[0]
        self.assertTrue(atleast_nsamples.all())

    def test_fuzzpred_colsum_with_mean_mdeg(self):
        model = FKNN(2, 2, 2)
        model.fit(Xtrain, Ytrain_, init_strat='means')
        preds = model.predict(Xtest)
        colsums = np.sum(preds, axis=0)
        atleast_nsamples = colsums <= Xtest.shape[0]
        self.assertTrue(atleast_nsamples.all())

    def test_predict_without_fit(self):
        self.model = FKNN(2, 2, 2)
        self.assertRaises(ValueError, self.model.predict, Xtest)


class TestsFit(unittest.TestCase):

    def setUp(self):
        self.model = FKNN(2, 2, 2)

    def test_invalid_properties(self):
        tset = [(-1,2,2), (0,2,2), (2,4,2), (2,-1,2), (2,2,-1), (2,2,1)]
        for params in tset:
            self.model = FKNN(*params)
            self.assertRaises(ValueError, self.model.fit, Xtrain, Ytrain_)

    def test_valid_properties(self):
        self.model = FKNN(3, 3, 3)
        self.model.fit(Xtrain, Ytrain_)

    def test_single_class(self):
        nlabels = Ytrain.shape[0]
        singlelabels = np.ones(nlabels, dtype=np.int32).reshape(nlabels, 1)
        self.assertRaises(ValueError, self.model.fit, Xtrain, singlelabels)

    def test_fit_with_wrong_metric(self):
        metrics = [0, -1, 4, 1.5, 0.1, 1.1]
        with self.assertRaises(ValueError, msg='Fit with invalid similarity'):
            for metric in metrics:
                FKNN(2, metric, 2).fit(Xtrain, Ytrain_)

    def test_fit_with_correct_metric(self):
        metrics = [1, 2, 3]
        for metric in metrics:
            FKNN(2, metric, 2).fit(Xtrain, Ytrain_)

    def test_with_invalid_init_strat(self):
        strats = [4, [], 'bla', '', '   ', 'COMPLETE', 'Mean']
        for strat in strats:
            try:
                FKNN(2, 2, 2).fit(Xtrain, Ytrain_, init_strat=strat)
                self.fail(f"Init memberships with unknown strat: {strat}")
            except ValueError:
                pass

    def test_with_valid_init_strat(self):
        strats = ['complete', 'means']
        for strat in strats:
            FKNN(2, 2, 2).fit(Xtrain, Ytrain_, init_strat=strat)


def accuracy(reals, predictions):
    n_hits = 0
    for real, pred in zip(reals, predictions):
        if int(real) == int(pred):
            n_hits += 1
    return n_hits/len(reals)



