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
N_MFS = N_INPUTS * FSET_SIZE
N_RULES = FSET_SIZE ** N_INPUTS

print(f"MAX_EPOCHS: {MAX_EPOCH}\t",
      f"FUZZY SUBSET SIZE: {FSET_SIZE}\n",
      f"RULES: {N_RULES}\t",
      f"FUZZY SUBSETS: {N_MFS}")
print()

epoch = 0
premise_mean = np.random.normal(0, 1.0, (N_MFS, 1))
premise_std = np.ones((N_MFS, 1))
premise = np.hstack((premise_mean, premise_std))

def half_forward_pass(x):
    inps = np.repeat(x, FSET_SIZE)
    args = (inps - premise[:, 0])/premise[:, 1]
    l1 = np.exp(-args)
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


sysmat = []
print('\nFinding consequent parameters...')
for x in X:
    l1, l2, l3 = half_forward_pass(x)
    weights = np.repeat(l3, N_INPUTS + 1)
    x_ = np.append(x, 1.0)
    aux = np.tile(x_, N_POINTS)
    sysmat.append(weights * aux)
sysmat = np.array(sysmat)
lse_result = lse.Matricial().solve(sysmat, Y)
consequents = np.reshape(lse_result, (N_POINTS, N_INPUTS + 1))
print('Training...')
error = 0
while epoch < MAX_EPOCH:
    print(f"Epoch {epoch + 1}")
    l5 = []
    print("Forwarding...")
    for x, y in zip(X, Y):
        l1, l2, l3 = half_forward_pass(x)
        x_ = np.append(x, 1.0)
        fi = consequents @ x_
        l4 = fi * l3
        l5.append(l4.sum())
        dEdO5 = -2*(y - l5[-1])
        dO4dOc = np.tile(l3, (3, 1)).T * x_
        dO4dO3 = fi
        denom = l2.sum() ** 2
        num = -l2 + l2.sum()
        dO3dO2 = num / denom
        # COmpute derivative of rules
        mfids = np.arange(N_MFS).reshape((N_INPUTS, FSET_SIZE))
        dO2dO1 = []
        for k, rule in enumerate(product(*mfids.tolist())):
            dOr = []
            for mfid in rule:
                dOr.append(l2[k] / l1[mfid])
            dO2dO1.append(dOr)
        dO2dO1 = np.array(dO2dO1)
        dO1da = 0
        dO1dc = 0
        consequents += dO4dOc * dEdO5
    epoch += 1


