import os
import numpy as np
import unittest

from .context import fuzzy
from fuzzy.pca import PCA



DIRNAME = os.path.dirname(__file__)
FILEPATH = os.path.join(DIRNAME, 'dataset_blob.txt')


class TestsPCA(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestsPCA, self).__init__(*args, **kwargs)
        dataset = np.loadtxt(FILEPATH)
        self.Xtrain = dataset[:350, :-1]

    def test_initialise(self):
        PCA(3).fit_transform(self.Xtrain)
