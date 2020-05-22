from .context import fuzzy
from fuzzy.logic import mfs
import unittest


class TestBellThree(unittest.TestCase):

    def setUp(self):
        self.x = 4
        self.a = 3
        self.b = 2
        self.c = 2

    def test_mem_degree(self):
        res = mfs.genbell(self.x, self.a, self.b, self.c)
        self.assertAlmostEqual(res, 0.835051546392, 12)

    def test_mem_degree_zerodivision(self):
        with self.assertRaises(ValueError):
            mfs.genbell(self.x, 0.0, self.b, self.c)
            self.fail('Zero division with zero spam')

    def test_zero(self):
        inputs = [[0, 2, 0, 0],
                  [1, 2, 0, 0],
                  [0, 2, 0, 4],
                  [1, 2, 3, 0],
                  [1, 2, 0, 4],
                  [0, 2, 3, 4]]
        outputs = [0.5, 0.5, 0.5, 0.9846153854, 0.5, 1/65]
        for inp, out in zip(inputs, outputs):
            mdeg = mfs.genbell(*inp)
            self.assertAlmostEqual(mdeg, out)

