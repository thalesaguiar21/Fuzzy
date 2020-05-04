import math
from collections import defaultdict
from itertools import product

from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

from fuzzy import lse
from fuzzy.logic import mfs


# Make test data
X_CRISP, Y = make_blobs()
X = normalize(X_CRISP, axis=1, norm='l1')
N_POINTS, N_INPUTS = X.shape

# Set constants
MAX_EPOCH = 10
FSET_SIZE = 3
STEP_SIZE = 1e-10
TOL = 0.2
N_MFS = N_INPUTS * FSET_SIZE
N_RULES = FSET_SIZE ** N_INPUTS
N_PARAMS = 2*N_MFS + N_RULES*(N_INPUTS + 1)

# CONSTANTS
A = 0
B = 1
C = 2

print(f"Max epochs: {MAX_EPOCH}\t\t"
      f"Subset size: {FSET_SIZE}\n"
      f"Rules: {N_RULES}\t\t"
      f"Subsets: {N_MFS}\n"
      f"Parameters: {N_PARAMS}\t\t"
      f"Step: {STEP_SIZE}\n")


def build_mfs(inputs, nmfs=2):
    feat_spams, mins = _find_feat_spam(inputs)
    means = _find_means(feat_spams, mins, nmfs)
    mfspam = np.repeat(feat_spams, nmfs)
    plato = np.ones(means.shape)
    params = np.array([plato, mfspam, means])
    return params.T


def _find_feat_spam(inputs):
    mins = np.min(inputs, axis=0)
    maxs = np.max(inputs, axis=0)
    inp_spaces = np.abs(maxs - mins)
    return inp_spaces, mins


def _find_means(featspam, mins, nmfs):
    stepsizes = featspam / (nmfs-1)
    stepsizesrow = stepsizes.reshape(1, len(stepsizes))
    steps = np.arange(nmfs).reshape(nmfs, 1)
    splits = steps @ stepsizesrow + mins
    return splits.T.reshape(-1)


def half_forward_pass(x):
    inps = np.repeat(x, FSET_SIZE)
    args = (inps - premises[:, A])/premises[:, C]
    l1 = 1 / (1 + np.abs(args)**(2*premises[:, B]))
    mfids = np.arange(N_MFS).reshape((N_INPUTS, FSET_SIZE))
    l2 = []
    for rule in product(*mfids.tolist()):
        prod = 1
        for mfid in rule:
            prod *= l1[mfid]
        l2.append(prod)
    total = np.sum(l2)
    l2 = np.array(l2)
    l3 = l2 / total
    return l1, l2, l3


premises = build_mfs(X, FSET_SIZE)
sysmat = []

print('\nFinding consequent parameters...')
for x in X:
    __, __, weights = half_forward_pass(x)
    weightcol = weights.reshape((N_RULES, 1))
    x_ = np.append(x, 1.0).reshape((1, N_INPUTS + 1))
    sys_products = weightcol @ x_
    sysline = sys_products.reshape(-1)
    sysmat.append(sysline)
sysmat = np.array(sysmat)
lse_result = lse.Matricial().solve(sysmat, Y)
consequents = np.reshape(lse_result, (N_RULES, N_INPUTS + 1))

print('Finding premise params...')
epoch = 0
error = 0
p = 0
premise_grad = np.zeros((N_MFS, 3))
converged = False
while epoch < MAX_EPOCH and not converged:
    print(f"Epoch {epoch + 1}", end='\r')
    o5 = []
    for x, y in zip(X, Y):
        o1, o2, o3 = half_forward_pass(x)
        x_ = np.append(x, 1.0)
        fi = consequents @ x_
        o4 = fi * o3
        o5.append(o4.sum())

        de_do5 = -2*(y - o5[-1])
        o2sum = np.sum(o2)
        do3_do2 = [(o2sum - o2i)/(o2sum**2) for o2i in o2]

        mfids = np.arange(N_MFS).reshape((N_INPUTS, FSET_SIZE))
        mfderivs = {id_: 1 for id_ in range(N_MFS)}
        i = 0
        for rule in product(*mfids):
            for mfid in rule:
                mfderivs[mfid] += o2[i] / o1[mfid]
            i += 1
        do2_do1 = np.array(list(mfderivs.values()))
        acum_do = de_do5 * fi * do3_do2

        xP = np.repeat(x, FSET_SIZE)
        _2b = 2*premises[:, B]
        xc = xP - premises[:, C]
        xca = np.abs(xc / premises[:, A])
        sqr_1_xca = (xca**_2b + 1) ** 2
        xca_b_1 = xca**(2*(premises[:, B] - 1))

        do1_da = _2b * xc**2 * xca_b_1
        do1_da /= premises[:, A]**3 * sqr_1_xca

        do1_db = 2*xca**(_2b) * np.log10(xca + 1e-10)
        do1_db /= sqr_1_xca

        do1_dc = _2b*xc * xca_b_1
        do1_dc /= premises[:, A]**2 * sqr_1_xca


    eta = STEP_SIZE / math.sqrt(np.sum(premise_grad**2))
    learnrate = -eta * premise_grad
    premises += (learnrate * premise_grad).T
    premises = np.fmax(premises, np.finfo(np.float64).eps)

    curr_sqerror = sum((Y - o5)**2)
    loss = abs(error - curr_sqerror)
    error = curr_sqerror
    converged = loss <= TOL
    print(f"Loss: {loss}")
    epoch += 1

