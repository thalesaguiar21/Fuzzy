from .context import fuzzy
import fuzzy.neural.anfis as anfis
import unittest


class TestSugeno(unittest.TestCase):

    def when_model_qtd_of_mf_is(self, qtd_mf):
        self.model = anfis.Sugeno(qtd_mf)

    def test_setup_arch(self):
        self.when_model_qtd_of_mf_is(3)
