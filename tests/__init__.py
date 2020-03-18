import os

import numpy as np


DIRNAME = os.path.dirname(__file__)
FILEPATH = os.path.join(DIRNAME, 'dataset_blob.txt')

_data = np.loadtxt(FILEPATH)
Xtrain = _data[:350, :-1]
Xtest = _data[350:, :-1]
Ytrain = _data[:350, -1].reshape(350, 1)
Ytest = _data[350:, -1].reshape(150, 1)

