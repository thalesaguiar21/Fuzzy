from .context import fuzzy
import fuzzy.models.fcm as fcm
import unittest
import numpy as np


class TestsFCM(unittest.TestCase):

    def setUp(self):
        self.mfcm = fcm.FCM(3, 2)
        dataset = np.loadtxt('/home/thales/DEV/Fuzzy/tests/dataset_blob.txt')
        self.data = dataset[:, :-1]
        self.labels = dataset[:, :-1]

    def test_partitions_dim(self):
        parts, _ = self.mfcm.fit(self.data, 0.2)
        self.assertEqual(parts.shape, (500, 3))

