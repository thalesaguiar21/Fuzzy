import numpy as np
from itertools import product
import anfys.lse as regression
from enum import Enum, auto


class Layer(Enum):
    FUZZYFIER = auto()
    FIRE = auto()
    NORMALIZER = auto()
    DEFUZZIFIER = auto()
    OUTPUT = auto()


def hybrid_online(anfis, entry, output):
    l1tol3 = _half_forward_pass(entry, output)
    l4 = _update_consequent_parameters(anfis, l1tol3, entry, output)
    l5 = _prediction(l4)
    return l5


def _half_forward_pass(anfis, entry, output):
    # Forward inputs until the third layer
    layer1 = _fuzzysets_membership_degrees(anfis, entry)
    layer2 = _rules_fire_strength(layer1)
    layer3 = _averaged_fire_strength(layer2)
    return layer1, layer2, layer3


def _fuzzysets_membership_degrees(anfis, inputs):
    l1size = anfis.l1size()
    layer1 = np.zeros((l1size, anfis.qtd_mfs))
    param_range = np.arange(0, l1size + 1, anfis.qtd_mfs)
    for feat, subset, out in zip(inputs, anfis.sets, range(l1size)):
        at, untill = param_range[out], param_range[out + 1]
        output = subset.evaluate(feat, anfis.prem_params[at:untill])
        layer1[out] = output
    return layer1


def _rules_fire_strength(anfis, mdegrees):
    nodes_id = np.arange(anfis.qtd_mfs)
    # Create every combination for the given
    layer2 = []
    for mf in product(nodes_id, repeat=anfis.qtd_inputs):
        rule = [mdegrees[n_set, mf[n_set]] for n_set in range(anfis.qtd_sets)]
        layer2.append(np.prod(rule))
    return np.array(layer2)


def _averaged_fire_strength(fire_strengths):
    total_strength = np.sum(fire_strengths)
    return [rstrength / total_strength for rstrength in fire_strengths]


def _update_consequent_parameters(anfis, layers, entry, output):
    weights = layers[Layer.NORMALIZER]
    if isinstance(anfis, ):
        _update_consequent_parameters(anfis, entry, output, weights)
    else:
        raise ValueError(
            'Unkown {} model passed to learn.update_parameters!'.format(anfis))


def _solve_consequent_system(anfis, entry, output, weights):
    column_weights = np.array([weights]).T
    coefs = column_weights.dot(entry)
    anfis.add_linsys_equation(coefs, output)
    anfis.cons_params = regression.solve(
        anfis.linsys_coefs, anfis.linsys_resul)


def _prediction(defuzzified_values):
    return np.sum(defuzzified_values)
