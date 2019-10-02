from .context import anfys
import anfys.fuzzy.operations.tnorm as tnorm
import unittest


class TestTnorm(unittest.TestCase):

    def set_up(self, a, b):
        self.a = a
        self.b = b

    def tear_down(self):
        self.a = 3
        self.b = 2

    def expect_tnorm(self, fun, a, b, ex):
        self.set_up(a, b)
        self.assertEqual(ex, fun(self.a, self.b))
        self.tear_down()

    def raising_tnorm(self, fun, a, b, error):
        try:
            self.set_up(a, b)
            fun(a, b)
        except error:
            pass
        else:
            self.tear_down()

    def test_fmin_correct(self):
        self.expect_tnorm(tnorm.fmin, 3, 3, 3)
        self.expect_tnorm(tnorm.fmin, 3, 2, 2)
        self.expect_tnorm(tnorm.fmin, 0, -1, -1)

    def test_fmin_none(self):
        self.raising_tnorm(tnorm.fmin, None, 3, ValueError)
        self.raising_tnorm(tnorm.fmin, None, None, ValueError)
        self.raising_tnorm(tnorm.fmin, 0, None, ValueError)

    def test_prod_correct(self):
        self.expect_tnorm(tnorm.prod, 3, 3, 9)
        self.expect_tnorm(tnorm.prod, 3, 2, 6)
        self.expect_tnorm(tnorm.prod, 0, -1, 0)

    def test_prod_none(self):
        self.raising_tnorm(tnorm.prod, None, 3, ValueError)
        self.raising_tnorm(tnorm.prod, None, None, ValueError)
        self.raising_tnorm(tnorm.prod, 0, None, ValueError)

    def test_lukas_correct(self):
        self.expect_tnorm(tnorm.lukasiewicz, 3, 3, 5)
        self.expect_tnorm(tnorm.lukasiewicz, 3, 2, 4)
        self.expect_tnorm(tnorm.lukasiewicz, 0, -1, 0)
        self.expect_tnorm(tnorm.lukasiewicz, 1, -1, 0)

    def test_lukas_none(self):
        self.raising_tnorm(tnorm.lukasiewicz, None, 3, ValueError)
        self.raising_tnorm(tnorm.lukasiewicz, None, None, ValueError)
        self.raising_tnorm(tnorm.lukasiewicz, 0, None, ValueError)

    def test_drastic_correct(self):
        self.expect_tnorm(tnorm.drastic, 1, 3, 3)
        self.expect_tnorm(tnorm.drastic, 3, 1, 3)
        self.expect_tnorm(tnorm.drastic, 0, 1, 0)
        self.expect_tnorm(tnorm.drastic, 1, 1, 1)
        self.expect_tnorm(tnorm.drastic, 3, 3, 0)
        self.expect_tnorm(tnorm.drastic, 1, -3, -3)

    def test_drastic_none(self):
        self.raising_tnorm(tnorm.drastic, None, 3, ValueError)
        self.raising_tnorm(tnorm.drastic, None, None, ValueError)
        self.raising_tnorm(tnorm.drastic, 0, None, ValueError)

    def test_nilpotent_correct(self):
        self.expect_tnorm(tnorm.nilpotent, 1, 1, 1)
        self.expect_tnorm(tnorm.nilpotent, 3, 1, 1)
        self.expect_tnorm(tnorm.nilpotent, 0, 1, 0)
        self.expect_tnorm(tnorm.nilpotent, 1, 1, 1)
        self.expect_tnorm(tnorm.nilpotent, 3, -3, 0)
        self.expect_tnorm(tnorm.nilpotent, 1, -3, 0)

    def test_nilpotent_none(self):
        self.raising_tnorm(tnorm.nilpotent, None, 3, ValueError)
        self.raising_tnorm(tnorm.nilpotent, None, None, ValueError)
        self.raising_tnorm(tnorm.nilpotent, 0, None, ValueError)

    def test_hamacher_correct(self):
        self.expect_tnorm(tnorm.hamacher, 0, 0, 0)
        self.expect_tnorm(tnorm.hamacher, 3, 1, 3)
        self.expect_tnorm(tnorm.hamacher, 0, 1, 0)
        self.expect_tnorm(tnorm.hamacher, 1, 0, 0)
        self.expect_tnorm(tnorm.hamacher, 3, -3, -1)
        self.expect_tnorm(tnorm.hamacher, 1, -3, -3)

    def test_hamacher_none(self):
        self.raising_tnorm(tnorm.hamacher, None, 3, ValueError)
        self.raising_tnorm(tnorm.hamacher, None, None, ValueError)
        self.raising_tnorm(tnorm.hamacher, 0, None, ValueError)
