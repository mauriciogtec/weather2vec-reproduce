import argparse
import os
import yaml
from os.path import join
import logging

import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA
from openTSNE import TSNE

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as vF
import pytorch_lightning as pl

import baselines


def make_patches(C, k, pad=False):
    if pad:
        C = F.pad(torch.FloatTensor(C), (k, k)).numpy()
    padded = C.transpose(0, 2, 3, 0)
    patches = extract_patches_2d(padded, (k, k))
    patches = patches.transpose(0, 3, 1, 2)
    return patches


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    method = args.method
    sim = args.sim
    output_dir = f"{args.output}/lu17/embeddings/{method}/{sim:03d}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # utils.set_seed(args.seed)
    pl.seed_everything(args.seed)
    dir = f"{args.sims_dir}"

    # read datase
    data = np.load("data_simstudy/lu2017.npz")
    savedir = f"{args.output}/lu17/{args.method}"
    
    Y = data['monthly_PM_std'].transpose(2, 1, 0)
    C = data['covars'].transpose(3, 0, 2, 1)  # ntxndxnrxnc
    C = (C - np.nanmean(C, (2, 3), keepdims=True)) / np.nanstd(C, (2, 3), keepdims=True)

    # make masks
    M = (~np.isnan(Y) & ~np.any(np.isnan(C), 1)).astype(np.float32)
    Y[np.isnan(Y)] = 0.0
    C[np.isnan(C)] = 0.0
    C, Y, M = [torch.FloatTensor(u) for u in (C, Y, M)]

    nt, nd, nr, nc = C.shape

    if method == "wx":
        model = baselines.WXRegression(nd, (6, 4), **vars(args))
    else:
        pass

    errors = []
    for t in range(nt):
        ix = (np.arange(nt) != t)
        dstrain = TensorDataset(C[ix], Y[ix], M[ix])
        dsval = TensorDataset(C[t, None], Y[t, None], M[t, None])
        dltrain = DataLoader(dstrain, pin_memory=True) 
        dlval = DataLoader(dsval, pin_memory=True)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            enable_progress_bar=args.verbose,
            max_epochs=args.epochs,
            logger=None
        )
        trainer.fit(model, train_dataloaders=dltrain, val_dataloaders=dlval)
        0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--sim", type=int, default=0)
    parser.add_argument("--output", type=str, default="results-lu17")
    parser.add_argument("--task", type=str, default="nonlinear")
    parser.add_argument("--sims_dir", type=str, default="simulations")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    avail_methods = ["wx"]
    parser.add_argument("--method", type=str, default="wx", choices=avail_methods)
    parser.add_argument("--silent", default=True, dest="verbose", action="store_false")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    main(args)