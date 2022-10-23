import numpy as np
import cv2
from scipy import signal
from itertools import product as prod
# import scipy
# import networkx as nx
# from scipy import sparse


def sample_gp2d(
    nr: int, nc: int, a: float = 1.0, b: float = 1.0,
) -> np.ndarray:
    K = np.zeros((nr, nc, nr, nc))
    rows = np.arange(nr)
    cols = np.arange(nc)
    X = np.array(list(prod(rows, cols, rows, cols)))
    D = X[:, 0:2] - X[:, 2:4]
    K = a * np.exp(-0.5 * (b ** 2) * np.square(D).sum(-1))
    K = K.reshape(nr * nc, nr * nc)
    x = np.random.multivariate_normal(np.zeros(nr * nc), K)
    x = x.reshape(nr, nc)
    x -= x.mean()
    x /= x.std()
    return x


def up(x: np.ndarray, factor: int):
    for _ in range(factor):
        x = cv2.pyrUp(x)
    return x

def center_crop(x: np.ndarray, pad: int):
    assert pad > 1
    return x[(pad):-(pad), (pad):-(pad)]


def local_potential(ksize: int):
    K = np.zeros((ksize, ksize))
    for i in range(ksize):
        for j in range(ksize):
            x = [i - ksize // 2, j - ksize // 2]
            K[i, j] = np.linalg.norm(x, 2) / ksize
    return K


def diff(X, axis):
    padw = [[0, 0] for _ in range(len(X.shape))]
    padw[axis] = [1, 0]
    X_ = np.pad(X, padw, mode='linear_ramp')
    return np.diff(X_, 1, axis)


def diff_change_pot(k: int, axis=0):
    assert k % 2 == 1
    out = np.zeros((k, k))
    if axis == 0:
        out[:(k//2), :] = -1
        out[(k//2 + 1):, :] = 1
    else:
        out[:, :(k//2)] = -1
        out[:, (k//2 + 1):] = 1
    return out


def potential_features(nr: int, nc: int, k: int, nonlinear: bool = False):
    P = sample_gp2d(nr // 4 + 1, nc // 4 + 1, b=4.0 / k)
    P = up(P, factor=2)
    # K = local_potential(ksize=k)
    # L = signal.correlate2d(P, K, mode="same")
    U = - center_crop(diff(P, 0), 2)
    V = - center_crop(diff(P, 1), 2)
    KU = diff_change_pot(k, 0)
    KV = diff_change_pot(k, 1)
    if nonlinear:
        tmpU = signal.correlate2d(np.sign(U), KU, mode='same')
        tmpV = signal.correlate2d(np.sign(V), KV, mode='same')
    else:
        tmpU = signal.correlate2d(U, KU, mode='same')
        tmpV = signal.correlate2d(V, KV, mode='same')
    L = tmpU + tmpV
    # L = center_crop(L, 2)
    P = center_crop(P, 2)
    return U, V, P, L


def sample_scattered_points(nr: int, nc: int, N: int):
    mask = np.zeros((nr * nc, ), dtype=bool)
    idx = np.random.choice(np.arange(nr * nc), N, replace=False)
    mask[idx] = True
    mask = mask.reshape(nr, nc)
    return mask


def nn_imputation(A: np.ndarray, obs_mask: np.ndarray):
    nr, nc = A.shape
    x, y = np.where(obs_mask)
    sample_list = np.stack([x, y], axis=1)
    X, Y = np.meshgrid(np.arange(nc), np.arange(nr))
    arr = np.stack([Y.flatten(), X.flatten()], axis=1)
    closest = np.argmin(np.linalg.norm(arr[:, None] - sample_list[None], axis=2), axis=1)
    ix = sample_list[closest]
    ix = ix[:, 0], ix[:, 1]
    nn = np.zeros((nr, nc), dtype=bool)
    nn[Y.flatten(), X.flatten()] = A[ix]
    return nn

