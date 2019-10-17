from .context import fuzzy
from fuzzy.cluster import FCM
import unittest
import numpy as np
import os

DIRNAME = os.path.dirname(__file__)
FILEPATH = os.path.join(DIRNAME, 'dataset_blob.txt')

class TestsFCM(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestsFCM, self).__init__(*args, **kwargs)
        dataset = np.loadtxt(FILEPATH)
        self.Xtrain = dataset[:350, :-1]
        self.Xtest = dataset[350:, :-1]
        self.Ytrain = dataset[:350, -1]
        self.Ytest = dataset[350:, -1]
        self.mfcm = FCM(nclusters=3, fuzzyness=2)
        self.mfcm.fit(self.Xtrain, 0.2)

    def setUp(self):
        pass

    def test_cluster_fuzzyness_out(self):
        self.assertRaises(ValueError, FCM, 3, -2)
        self.assertRaises(ValueError, FCM, 3, -1)
        self.assertRaises(ValueError, FCM, 3, 0)
        self.assertRaises(ValueError, FCM, 3, 0.999)

    def test_ncluster_out(self):
        self.assertRaises(ValueError, FCM, 1, 2)
        self.assertRaises(ValueError, FCM, 0.1, 2)
        self.assertRaises(ValueError, FCM, 0, 2)

    def test_fuzzyness_in(self):
        self.assertNotRaises(ValueError, FCM, 3, 2)
        self.assertNotRaises(ValueError, FCM, 3, 2.001)
        self.assertNotRaises(ValueError, FCM, 3, 3)

    def test_ncluster_in(self):
        self.assertNotRaises(ValueError, FCM, 2, 2)
        self.assertNotRaises(ValueError, FCM, 3, 2)

    def assertNotRaise(self, func, **kwargs):
        try:
            func(**kwargs)
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
            self.mfcm.predict_fuzz(self.Xtest)
        except:
            self.fail()

    def test_predict_dim(self):
        preds = self.mfcm.predict(np.array([[1, 2], [2, 3]]))
        self.assertEqual(preds.shape, (2, ))

    def test_predict_testset(self):
        try:
            self.mfcm.predict(self.Xtest)
        except:
            self.fail()

