from .context import fuzzy
import fuzzy.logic.tconorm as tconorm
import unittest


class TestTconorm(unittest.TestCase):

    def set_up(self, a, b):
        self.a = a
        self.b = b

    def tear_down(self):
        self.a = 3
        self.b = 2

    def expect_tconorm(self, fun, a, b, ex):
        self.set_up(a, b)
        self.assertEqual(ex, fun(self.a, self.b))
        self.tear_down()

    def raising_tconorm(self, fun, a, b, error):
        try:
            self.set_up(a, b)
            fun(a, b)
        except error:
            pass
        else:
            self.tear_down()

    def test_fmax_correct(self):
        self.expect_tconorm(tconorm.fmax, 3, 3, 3)
        self.expect_tconorm(tconorm.fmax, 3, 2, 3)
        self.expect_tconorm(tconorm.fmax, 0, -1, 0)

    def test_prob_correct(self):
        self.expect_tconorm(tconorm.probabilistic_sum, 3, 3, -3)
        self.expect_tconorm(tconorm.probabilistic_sum, 2, 2, 0)
        self.expect_tconorm(tconorm.probabilistic_sum, 2, 0, 2)
        self.expect_tconorm(tconorm.probabilistic_sum, 0, 2, 2)

    def test_bounded_correct(self):
        self.expect_tconorm(tconorm.bounded_sum, 3, 3, 1)
        self.expect_tconorm(tconorm.bounded_sum, 3, -3, 0)
        self.expect_tconorm(tconorm.bounded_sum, 3, -2.9999999, 3 - 2.9999999)
        self.expect_tconorm(tconorm.bounded_sum, 0, 1.0000001, 1.0)

    def test_drastic_correct(self):
        self.expect_tconorm(tconorm.drastic, 0, 3, 3)
        self.expect_tconorm(tconorm.drastic, 3, 0, 3)
        self.expect_tconorm(tconorm.drastic, 0, 0, 0)
        self.expect_tconorm(tconorm.drastic, 1, 1, 1.0)
        self.expect_tconorm(tconorm.drastic, 3, -3, 1.0)

    def test_nilpotent_correct(self):
        self.expect_tconorm(tconorm.nilpotent_max, -1, 1, 1)
        self.expect_tconorm(tconorm.nilpotent_max, 3, -2.0000001, 3)
        self.expect_tconorm(tconorm.nilpotent_max, 0, 1, 1)
        self.expect_tconorm(tconorm.nilpotent_max, 1, 0, 1)

    def test_einstein_correct(self):
        self.expect_tconorm(tconorm.einstein_sum, 0, 0, 0)
        self.expect_tconorm(tconorm.einstein_sum, 0, 1, 1.0)
        self.expect_tconorm(tconorm.einstein_sum, 1, 0, 1.0)
        self.expect_tconorm(tconorm.einstein_sum, 1, 1e-10, 1.0)

