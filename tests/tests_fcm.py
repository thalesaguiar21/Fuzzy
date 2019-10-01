from .context import fuzzy
from fuzzy.models.clustering import FCM
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
        self.mfcm = FCM(3, 2)
        self.mfcm.fit(self.Xtrain, 0.2)

    def setUp(self):
        pass

    def test_cluster_fuzzyness_out(self):
        self.assertRaises(ValueError, FCM, 3, -2)
        self.assertRaises(ValueError, FCM, 3, -1)
        self.assertRaises(ValueError, FCM, 3, 0)
        self.assertRaises(ValueError, FCM, 3, 0.999)

    def test_ncluster_out(self):
        self.assertRaises(ValueError, FCM, -1, 2)
        self.assertRaises(ValueError, FCM, 0, 2)
        self.assertRaises(ValueError, FCM, 0.1, 2)

    def test_fuzzyness_in(self):
        self.assertNotRaise(FCM, 3, 1)
        self.assertNotRaise(FCM, 3, 1.001)
        self.assertNotRaise(FCM, 3, 2)

    def test_ncluster_in(self):
        self.assertNotRaise(FCM, 1, 2)
        self.assertNotRaise(FCM, 2, 1)

    def assertNotRaise(self, func, *args):
        try:
            func(*args)
        except:
            self.fail()

    def test_partitions_dim(self):
        parts = self.mfcm.partitions
        self.assertEqual(parts.shape, (350, 3))

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

