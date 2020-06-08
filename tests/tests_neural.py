import unittest

from .context import fuzzy
from fuzzy.neural import anfis

from . import Xtrain, Xtest, Ytrain, Ytest


class TestsAnfis(unittest.TestCase):

    def setUp(self):
        self.fis = anfis.ANFIS()

    def test_fit_strat_error(self):
        strats = ['ONLINE', 'OFFline', 'OFFLINE', 'onLINE', 'tests', '', 123]
        for strat in strats:
            with self.assertRaises(ValueError, msg=f"for '{strat}'"):
                self.fis.fit(Xtrain, Ytrain, strat)

    def test_fit_known(self):
        strats = ['online', 'offline']
        for strat in strats:
            self.fis.fit(Xtrain, Ytrain, strat)


