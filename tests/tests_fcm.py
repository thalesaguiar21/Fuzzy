from .context import fuzzy
import fuzzy.models.fcm as fcm
import unittest
import numpy
import sklearn.datasets as skdata


class TestsFCM(unittest.TestCase):

    def setUp(self):
        self.mfcm = fcm.FCM(3, 2)
        points, labels = skdata.make_blobs(500, 2, 3)
        self.data = points
        self.labels = labels

    def test_partitions_dim(self):
        parts, _ = self.mfcm.fit(self.data, 0.2)
        self.assertEqual(parts.shape, (500, 3))
