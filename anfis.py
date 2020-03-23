import math
from collections import defaultdict
from itertools import product

from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
import numpy as np

from fuzzy import lse


# Make test data
X_CRISP, Y = make_blobs()
X = normalize(X_CRISP, axis=1, norm='l1')
N_POINTS, N_INPUTS = X.shape

MAX_EPOCH = 10
FSET_SIZE = 10
STEP_SIZE = 1e-10
TOL = 0.2
N_MFS = N_INPUTS * FSET_SIZE
N_RULES = FSET_SIZE ** N_INPUTS
N_PARAMS = 2*N_MFS + N_RULES*(N_INPUTS + 1)

print(f"Max epochs: {MAX_EPOCH}\t\t"
      f"Subset size: {FSET_SIZE}\n"
      f"Rules: {N_RULES}\t\t"
      f"Subsets: {N_MFS}\n"
      f"Parameters: {N_PARAMS}\t\t"
      f"Step: {STEP_SIZE}\n")



def half_forward_pass(x):
    inps = np.repeat(x, FSET_SIZE)
    args = (inps - premise[:, 0])/premise[:, 1]
    l1 = np.exp(-args**2)
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


premise_mean = np.random.normal(0, 1.0, (N_MFS, 1))
premise_std = np.ones((N_MFS, 1))
premise = np.hstack((premise_mean, premise_std))
sysmat = []

print('\nFinding consequent parameters...')
for x in X:
    __, __, l3 = half_forward_pass(x)
    weights = np.repeat(l3, N_INPUTS + 1)
    x_ = np.append(x, 1.0)
    aux = np.tile(x_, N_POINTS)
    breakpoint()
    sysmat.append(weights * aux)
sysmat = np.array(sysmat)
lse_result = lse.Matricial().solve(sysmat, Y)
consequents = np.reshape(lse_result, (N_POINTS, N_INPUTS + 1))

print('Finding premise params...')
epoch = 0
error = 0
p = 0
premise_grad = np.zeros((2, N_MFS))
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
        mfids = np.arange(N_MFS).reshape((N_INPUTS, FSET_SIZE))
        mfderivs = {id_: 1 for id_ in range(N_MFS)}
        i = 0
        for rule in product(*mfids):
            for mfid in rule:
                mfderivs[mfid] *= fi[i] * o2[i] / o1[mfid]
            i += 1
        do2_do1 = np.array(list(mfderivs.values()))

        xpropagated = np.repeat(x, FSET_SIZE) 
        expderiv_c = (xpropagated-premise[:, 0]) / premise[:, 1]**2
        do1_dc = 2*(expderiv_c) * o1
        expderiv_a = (xpropagated-premise[:, 0])**2 / premise[:, 1]**3
        do1_da = 2*(expderiv_a) * o1

        grad_aux = de_do5 * do2_do1
        premise_grad[0] += grad_aux * do1_dc
        premise_grad[1] += grad_aux * do1_da


    eta = STEP_SIZE / math.sqrt(np.sum(premise_grad**2))
    learnrate = -eta * premise_grad
    premise += (learnrate * premise_grad).T
    premise = np.fmax(premise, np.finfo(np.float64).eps)

    curr_sqerror = sum((Y - o5)**2)
    loss = abs(error - curr_sqerror)
    error = curr_sqerror
    converged = loss <= TOL
    print(f"Loss: {loss}")
    epoch += 1

