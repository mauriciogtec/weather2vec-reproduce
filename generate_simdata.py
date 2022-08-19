import numpy as np
import argparse
from scipy.special import expit
from scipy.stats import norm
import os
from os.path import join
import proplot as pplt
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from utils import set_seed


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


def main(args: argparse.Namespace):
    plt.rc("text", usetex=True)
    set_seed(args.seed)
    nsims = args.nsims
    # load covariates
    nr, nc, ksize = 128, 256, args.ksize

    os.makedirs(args.output_dir, exist_ok=True)
    with open(join(args.output_dir, "args.yaml"), "w") as io:
        yaml.dump(vars(args), io)

    num_plots = args.nplots
    fig, axs = pplt.subplots(nrows=num_plots, ncols=3, wspace=0.1, span=False)

    for i in tqdm(range(nsims), disable=(not args.verbose)):
        U, V, P, L = potentials.potential_features(nr, nc, ksize, nonlinear=args.nonlinear)
        C = np.stack([U, V], 0)
        L_normed = (L - L.mean()) / L.std()
        logits = L_normed # 0.5 * (L_normed - 1)
        if args.spatial:
            U2 = potentials.sample_gp2d(nr//4, nc//4, b=8.0 / ksize)
            U2 = potentials.up(U2, factor=2)
            logits = np.sqrt(0.5) * (logits + U2)
        # outcome_mean -= outcome_mean.mean()
        # outcome_mean /= outcome_mean.std()
        effect_size = 0.1

        for dset in ("train", "test"):
            # for t in rangse(nt):
            # apply kernelss
            prob = expit(logits)

            # make spatial unif noise in (0,1)
            # swt = 0.1 if args.spatial else 0.0
            swt = 0.0
            U1 = np.random.rand(nr, nc)
            if swt > 0.0:
                U2 = potentials.sample_gp2d(nr//4, nc//4, b=8.0 / ksize)
                U2 = potentials.up(U2, factor=2)
                U2 = norm.cdf(U2)
            else:
                U2 = 0.0
            U = (1.0 - swt) * U1 + swt * U2
            assignment = U < prob

            swt = 0.5
            eps0 = np.random.normal(size=(nr, nc))
            eps1 = potentials.sample_gp2d(nr//4, nc//4, b=4.0 / ksize)
            eps1 = potentials.up(eps1, factor=2)
            eps = (1.0 - swt) * eps0 + swt * eps1
            outcome = - np.sqrt(0.5) * L_normed + np.sqrt(0.5) * eps + effect_size * assignment

            if i < num_plots and dset == "train":
                num = i
                m0 = axs[num, 0].imshow(
                    prob, cmap="greys_r", vmin=0.0, vmax=1.0
                )
                axs[num, 0].axis("off")
                axs[num, 0].format(ylabel="Example 1")
                m1 = axs[num, 1].imshow(assignment, cmap="greys_r")
                axs[num, 1].axis("off")
                m2 = axs[num, 2].imshow(outcome, cmap="greys_r")
                axs[num, 2].axis("off")

            np.savez(
                join(args.output_dir, f"{i:03d}_{dset}.npz"),
                covars=C,
                potential=P,
                outcome=outcome,
                prob=prob,
                assignment=assignment,
                effect_size=effect_size,
            )

    fig.colorbar(
        m0, loc="t", length=0.75, label="treatment probability", col=1
    )
    # els = [
    #     Patch(facecolor="black", edgecolor="grey", label="Treatment"),
    #     Patch(facecolor="white", edgecolor="grey", label="No treatement"),
    # ]
    # plt.legend(m1, els, loc="t", frame=False, ncol=2)
    fig.colorbar(
        m1,
        loc="t",
        length=0.75,
        label="treatment assignment",
        ticks=[0.0, 1.0],
        col=2,
    )
    fig.colorbar(m2, loc="t", length=0.75, label="simulated outcome", col=3)
    fig_path = join(args.output_dir, "examples.png")
    fig.format(xlabel="longitude", ylabel="latitude")
    fig.save(fig_path, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=110104)
    parser.add_argument("--nsims", type=int, default=30)
    parser.add_argument("--nplots", type=int, default=3)
    parser.add_argument("--ksize", type=int, default=9)
    parser.add_argument("--output_dir", type=str, default="simulations/test_potentials")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--nonlinear", default=False, action="store_true")
    parser.add_argument("--spatial", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
