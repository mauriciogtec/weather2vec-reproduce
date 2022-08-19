#%%
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import seaborn as sns
import pandas as pd
from itertools import product as prod
import proplot as pplt
from collections import defaultdict
from scipy.special import expit
from tqdm import tqdm
import itertools as it
from torch import nn
import argparse
import scipy
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import sys
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from patsy import dmatrix
from models import UNetEncoder, Decoder, MeanOnlyDecoder
import torch
import torchvision.transforms.functional as vF
import numpy as np
# from sksparse.cholmod import cholesky
from scipy import sparse
from utils import inv_perm, nbrs_avg, load_training_data


yearly_average = True
nd, nr, nc = 5, 128, 256
if yearly_average:
    nd *= 2
ksize = 13
dev = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(110104)

# %%

C, names, Yfull, M = load_training_data(
    path="data/training_data.pkl",
    standardize_so4=True,log_so4=True,
    year_averages=yearly_average
)
_, _, Yraw, Mraw, locs = load_training_data(path="data/training_data.pkl", return_pp_data=True)

r, c = np.where(M[0])
Y = Yfull[:, locs.row.values, locs.col.values]
Y.shape
print(locs.shape)
locs.head()

# %%

prefix = "h"
dirs = {
    "r1": f"./results/unsupervised-narr/{prefix}1_w2vec",
    "r3": f"./results/unsupervised-narr/{prefix}3_w2vec",
    "r5": f"./results/unsupervised-narr/{prefix}5_w2vec",
    "r7": f"./results/unsupervised-narr/{prefix}7_w2vec",
    "r9": f"./results/unsupervised-narr/{prefix}9_w2vec",
}

# %%

def radius_from_dir(s: str, prefix: str):
    return int(s.split("/")[-1].split("_")[0].replace(prefix, ""))


# %%

D = dict()
print("Loading models...")

for name, datadir in dirs.items():
    radius = radius_from_dir(datadir, prefix)
    args = argparse.Namespace()
    with open(os.path.join(datadir, "args.yaml"), "r") as io:
        for k, v in yaml.load(io, Loader=yaml.FullLoader).items():
            setattr(args, k, v)
            if k == "nbrs_av":
                setattr(args, "av_nbrs", v)
            elif k == "av_nbrs":
                setattr(args, "nbrs_av", v)
    bn_type ="frn" if not hasattr(args, "bn_type") else args.bn_type
    mkw = dict(
        n_hidden=args.nhidden,
        depth=args.depth,
        num_res=args.nres,
        ksize=args.ksize,
        groups=args.groups,
        batchnorm=True,
        batchnorm_type=bn_type,
    )
    print(vars(args))
    dkw = dict(batchnorm=True, offset=True, batchnorm_type=bn_type)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.local and args.nbrs_av == 0:
        enc = UNetEncoder(nd, args.nhidden, **mkw).to(dev)
        dec = Decoder(args.nhidden, nd, args.nhidden, **dkw).to(dev)
    else:
        enc = nn.Identity()
        dec = Decoder(nd, nd, args.nhidden, **dkw).to(dev)
    mod = nn.ModuleDict({"enc": enc, "dec": dec})
    objs = dict(
        mod=mod,
        args=args,
        radius=radius,
        nbrs_av=args.nbrs_av,
        local=args.local,
    )
    mod.eval()
    for p in mod.parameters():
        p.requires_grad = False
    mod = mod.to(dev)
    D[datadir] = objs

# %%

path = "./data/training_data.pkl"
# dataset = NARRData(path, 10)
# loader = DataLoader(dataset, batch_size=1, shuffle=False)
C, _, *_, locs = load_training_data(path, standardize_weather=True, return_pp_data=True, year_averages=yearly_average)
Corig, *_ = load_training_data(path, standardize_weather=False, year_averages=yearly_average)

t = (2004 - 2000) * 12 + 6
C = torch.FloatTensor(C).to(dev)
ix = range(t, t + 1)
Ct = C[ix]

row = locs.row.values
col = locs.col.values
locs.head()


# %%

os.makedirs(f"results/ex1_extraction/", exist_ok=True)

dfcols = []
for k, v in D.items():
    mod = v["mod"]
    mod.load_state_dict(torch.load(os.path.join(k, "model.pt")))
    with torch.no_grad():
        Z = mod["enc"](Ct)
        Z = Z.mean(0).cpu().numpy()
        Zmat = Z[:, row, col].T
        colnames = [f"C{i:02d}" for i in range(Zmat.shape[-1])]
        Z = pd.DataFrame(Zmat, columns=colnames)
        Z = pd.DataFrame(Zmat, columns=[x + f"_{len(dfcols)}" for x in colnames])
        dfcols.append(Z)
        Z = pd.concat([locs, Z], axis=1)
        Z.to_csv(f"results/ex1_extraction/{prefix}_{v['radius']:02d}.csv", index=False)

Cloc = Corig[ix].mean(0)[:, row, col].T
Z = pd.DataFrame(Cloc, columns=colnames[:Cloc.shape[1]])
Z = pd.concat([locs, Z], axis=1)
Z.to_csv(f"results/ex1_extraction/00.csv", index=False)   

dfcols = pd.concat(dfcols, axis=1)
dfcols = pd.concat([Z, dfcols], axis=1)
dfcols.to_csv(f"results/ex1_extraction/{prefix}_all.csv", index=False)
