import unittest

import numpy as np
from sklearn.metrics import accuracy_score

from .context import fuzzy
from fuzzy.cluster import FCM

from . import Xtrain, Xtest, Ytrain, Ytest


class TestsFCM(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestsFCM, self).__init__(*args, **kwargs)
        self.mfcm = FCM(nclusters=3, fuzzyness=2)
        self.mfcm.fit(Xtrain, 0.2)

    def tearDown(self):
        self.mfcm.nclusters = 3
        self.mfcm.fuzzyness = 2

    def setFCM(self, nclusters, m):
        self.mfcm.nclusters = nclusters
        self.mfcm.m = m

    def set_fit_raising(self, nclusters, m, error):
        self.setFCM(nclusters, m)
        self.assertRaises(error, self.mfcm.fit, Xtrain, 0.2)

    def set_fit_not_raising(self, nclusters, m):
        self.setFCM(nclusters, m)
        self.assertNotRaise(self.mfcm.fit, Xtrain, 0.2)

    def test_cluster_fuzzyness_out(self):
        self.set_fit_raising(3, -2, ValueError)
        self.set_fit_raising(3, -1, ValueError)
        self.set_fit_raising(3, -0, ValueError)
        self.set_fit_raising(3, -0.999, ValueError)

    def test_ncluster_out(self):
        self.set_fit_raising(1, 2, ValueError)
        self.set_fit_raising(0.1, 2, ValueError)
        self.set_fit_raising(0, 2, ValueError)

    def test_fuzzyness_in(self):
        self.set_fit_not_raising(3, 2)
        self.set_fit_not_raising(3, 2.001)
        self.set_fit_not_raising(3, 3)

    def test_ncluster_in(self):
        self.set_fit_not_raising(2, 2)
        self.set_fit_not_raising(3, 2)

    def assertNotRaise(self, func, *args):
        try:
            func(*args)
        except:
            self.fail()

    def test_partitions_dim(self):
        parts = self.mfcm.partitions
        self.assertEqual(parts.shape, (350, 3))

    def test_partitions_total_prob(self):
        parts = self.mfcm.partitions
        total_probs = np.sum(parts, axis=1)
        for prob in total_probs:
            self.assertAlmostEqual(prob, 1.0)

    def test_partition_individual_prob(self):
        for line in self.mfcm.partitions:
            for part in line:
                self.assertGreaterEqual(part, 0.0)
                self.assertLessEqual(part, 1.0)

    def test_partition_column_len(self):
        col_prob_sum = np.sum(self.mfcm.partitions, axis=0)
        for col_prob in col_prob_sum:
            self.assertGreater(col_prob, 0)
            self.assertLess(col_prob, self.mfcm.partitions.shape[0])

    def test_centroids_dim(self):
        centroids = self.mfcm.centroids
        self.assertEqual(centroids.shape, (3, 2))

    def test_predfuzz_dim(self):
        preds = self.mfcm.predict_fuzz(np.array([[1, 2], [2, 3]]))
        self.assertEqual(preds.shape, (2, 3))

    def test_predfuzz_none(self):
        preds = self.mfcm.predict_fuzz(None)
        self.assertEqual(preds, [])

    def test_predfuzz_empty(self):
        preds = self.mfcm.predict_fuzz([])
        self.assertEqual(preds, [])

    def test_predfuzz_testset(self):
        try:
            self.mfcm.predict_fuzz(Xtest)
        except:
            self.fail()

    def test_predict_dim(self):
        preds = self.mfcm.predict(np.array([[1, 2], [2, 3]]))
        self.assertEqual(preds.shape, (2, ))

    def test_predict_testset(self):
        try:
            self.mfcm.predict(Xtest)
        except:
            self.fail()

    def test_accuracy(self):
        preds = self.mfcm.predict(Xtrain)
        accuracy = accuracy_score(preds, Ytrain)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLess(accuracy, 1.0)


