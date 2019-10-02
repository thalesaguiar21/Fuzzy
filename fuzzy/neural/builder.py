import numpy as np
from anfys.fuzzy.subsets import FuzzySet


def configure_model(anfis, qtd_inputs, stdev=1.0):
    anfis.qtd_rules = qtd_inputs ** anfis.subset_size
    anfis.qtd_inputs = qtd_inputs
    _build_subsets(anfis, qtd_inputs)
    _build_prem_params(anfis, stdev)
    _initialise_cons_params(anfis, qtd_inputs)


def _build_subsets(anfis, qtd_inputs):
    anfis.fuzzysets = [FuzzySet(anfis.prem_mf) for _ in range(qtd_inputs)]


def _build_prem_params(anfis, stdev=1.0):
    stdevs = np.ones(anfis.l1_size()) * stdev
    means = np.linspace(-1.0, 1.0, anfis.qtd_mfs)
    means = np.array(means.tolist() * anfis.qtd_inputs)
    anfis.prem_params = np.vstack((stdevs, means)).T


def _initialise_cons_params(anfis, qtd_inputs):
    anfis.cons_params = np.zeros((anfis.qtd_rules, qtd_inputs))
