from datasets.stl10 import load_test_data
from scipy.linalg import toeplitz, pinv, inv
import numpy as np
from skimage.io import imsave

X, _ = load_test_data()
F = X[0]
imsave('clean.jpg', F)

l = 10
col = np.zeros(F.shape[1])
col[0] = 1/l
row = np.zeros(len(col) + l - 1)
row[0:l] = 1/l
H = toeplitz(col, row)

G = np.matmul(F, H)
imsave('noisy.jpg', G)

for l in range(1, 15):
    col = np.zeros(F.shape[1])
    col[0] = 1 / l
    row = np.zeros(len(col) + l - 1)
    row[0:l] = 1 / l
    H = toeplitz(col, row)
    denoised = np.matmul(pinv(H), G)
    imsave('denoised%s.jpg' % l, denoised)