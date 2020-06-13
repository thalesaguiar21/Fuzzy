import unittest

import numpy as np
from sklearn.metrics import accuracy_score

from .context import fuzzy
from fuzzy.cluster import fcm

from . import Xtrain, Xtest, Ytrain, Ytest


class TestsFCM(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestsFCM, self).__init__(*args, **kwargs)
        self.mfcm = fcm.FCM(nclusters=3, fuzzyness=2)
        self.mfcm.fit(Xtrain, 0.2)

    def setFCM(self, nclusters, m, tol, metric):
        self.mfcm.nclusters = nclusters
        self.mfcm.m = m
        self.mfcm.tol = tol
        self.p = metric

    def set_fit_raising(self, nclusters, m, error, metric):
        self.setFCM(nclusters, m, 0.2, metric)
        self.assertRaises(error, self.mfcm.fit, Xtrain)

    def set_fit_not_raising(self, nclusters, m, metric):
        self.setFCM(nclusters, m, 0.2, metric)
        self.assertNotRaise(self.mfcm.fit, Xtrain)

    def test_cluster_fuzzyness_out(self):
        self.set_fit_raising(3, -2, ValueError, 2)
        self.set_fit_raising(3, -1, ValueError, 2)
        self.set_fit_raising(3, -0, ValueError, 2)
        self.set_fit_raising(3, -0.999, ValueError, 2)

    def test_ncluster_out(self):
        self.set_fit_raising(1, 2, ValueError, 2)
        self.set_fit_raising(0.1, 2, ValueError, 2)
        self.set_fit_raising(0, 2, ValueError, 2)

    def test_fuzzyness_in(self):
        self.set_fit_not_raising(3, 2, 1)
        self.set_fit_not_raising(3, 2.001, 1)
        self.set_fit_not_raising(3, 3, 1)
        self.set_fit_not_raising(3, 2, 2)
        self.set_fit_not_raising(3, 2.001, 2)
        self.set_fit_not_raising(3, 3, 2)
        self.set_fit_not_raising(3, 2, 3)
        self.set_fit_not_raising(3, 2.001, 3)
        self.set_fit_not_raising(3, 3, 3)

    def test_ncluster_in(self):
        self.set_fit_not_raising(2, 2, 1)
        self.set_fit_not_raising(3, 2, 1)
        self.set_fit_not_raising(3, 2, 2)
        self.set_fit_not_raising(3, 2, 2)
        self.set_fit_not_raising(3, 2, 3)
        self.set_fit_not_raising(3, 2, 3)

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


class TestsSemiSupervisedFCM(unittest.TestCase):

    def test_part_rowsum(self):
        ss_fcm = fcm.SemiSupervisedFCM(3, 2)
        parts, __ = ss_fcm.fit(Xtrain, Ytrain)
        error = np.abs(np.sum(parts, axis=1) - 1)
        self.assertTrue(np.all(error <= 1e-7))

    def test_part_colsum(self):
        ss_fcm = fcm.SemiSupervisedFCM(3, 2)
        parts, __ = ss_fcm.fit(Xtrain, Ytrain)
        lower_than_samples = np.sum(parts, axis=0) <= Xtrain.shape[0]
        self.assertTrue(np.all(lower_than_samples))

    def test_ncenters_equal_nclass(self):
        ss_fcm = fcm.SemiSupervisedFCM(3, 2)
        __, centres = ss_fcm.fit(Xtrain, Ytrain)
        self.assertEqual(centres.shape[0], np.unique(Ytrain).size)

    def test_predict_with_labels(self):
        ss_fcm = fcm.SemiSupervisedFCM(3, 2)
        ss_fcm.fit(Xtrain, Ytrain)
        preds = ss_fcm.predict(Xtest)

    def test_accuracy(self):
        ss_fcm = fcm.SemiSupervisedFCM(3, 2)
        ss_fcm.fit(Xtrain, Ytrain)
        preds = ss_fcm.predict(Xtest)
        accuracy = accuracy_score(preds, Ytest)
        self.assertGreater(accuracy, 0.0)
        self.assertLess(accuracy, 1.0)

        
