import os
import sys
from typing import Optional
import numpy as np
import random
import torch
import torch.nn.functional as F
import logging
import pickle


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_logger(
    name: str = __name__,
    file: Optional[str] = None,
    print: bool = False,
    level=logging.DEBUG,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt: str = "%d-%b-%y %H:%M:%S",
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    if file is not None:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            os.remove(file)
        fh = logging.FileHandler(file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if print:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def nbrs_avg(C: torch.Tensor, radius: int, ord: int = 2):
    k = 2 * radius + 1
    ker = np.zeros((k, k))
    for i_ in range(k):
        for j_ in range(k):
            offset = [i_ - radius, j_ - radius]
            rad = np.linalg.norm(offset, ord=ord)
            if rad <= radius:
                ker[i_, j_] = 1.0
    ker /= ker.sum()
    box = torch.FloatTensor(ker).to(C.device)
    box = box[None, None, ...].repeat(C.size(1), 1, 1, 1)
    pad = (k // 2) if (k % 2 == 1) else (k // 2 - 1)
    out = F.conv2d(C, box, padding=pad, groups=C.size(1))
    return out


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: dict):
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


def inv_perm(perm: np.ndarray):
    s = np.empty_like(perm)
    s[perm] = np.arange(len(perm))
    return s


def load_training_data(
    path: str,
    standardize_weather: bool = False,
    standardize_so4: bool = False,
    log_so4: bool = False,
    remove_zeros: bool = True,
    return_pp_data: bool = False,
    year_averages: bool = False
):
    with open(path, "rb") as io:
        data = pickle.load(io)
    C = data["covars_rast"] # [:, weather_cols]
    names = data["covars_names"]
    if standardize_weather:
        C -= C.mean((0, 2, 3), keepdims=True)
        C /= C.std((0, 2, 3), keepdims=True)
    if year_averages:
        Cyearly_average = np.zeros_like(C)
        for t in range(C.shape[0]):
            if t < 12:
                Cyearly_average[t] = np.mean(C[:12], 0)
            else:
                Cyearly_average[t] = np.mean(C[(t - 12) : t], 0)
        C = np.concatenate([C, Cyearly_average], 1)
        names = names + [x + ".yavg" for x in names]
        names = [x.replace(".", "_") for x in names]

    Y = data["so4_rast"]
    M = data["so4_mask"]
    M[92:, 185:] = 0.0  # annoying weird corner
    M[80:, :60] = 0.0  # annoying weird corner
    if remove_zeros:
        M = (Y > 0) * M
        M = M * np.prod(M, 0)
    else:
        M = np.stack([M] * Y.shape[0])
    if log_so4:
        # Y = np.log(M * Y + 1e-8)
        Y = np.log(M * Y + 1.0)
    if standardize_so4:
        ix = np.where(M)
        Y -= Y[ix].mean()
        Y /= Y[ix].std()
    
    if not return_pp_data:
        return C, names, Y, M
    else:
        return C, names, Y, M, data["pp_locs"]


def yearly(x):
    N = len(x) // 12 + int(len(x) % 12 > 0)
    out = np.zeros(N + 1)
    for i in range(N):
        out[i] = x[(i * 12):min((i + 1) * 12, len(x))].mean()
    out[-1] = out[-2]
    return out