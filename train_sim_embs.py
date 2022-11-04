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
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms.functional as vF
import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger

import baselines


def rand_offset_target(Xt, radius):
    sgn = np.random.choice([-1, 1], replace=True, size=2)
    relpos = np.random.randint(0, radius + 1, size=2)
    relpos *= sgn
    Xt1 = vF.affine(Xt, translate=list(relpos), angle=0.0, scale=1.0, shear=0.0, fill=0.0)
    mask = vF.affine(torch.ones_like(Xt), translate=list(relpos), angle=0.0, scale=1.0, shear=0.0, fill=0.0)
    relpos = torch.FloatTensor(relpos).to(Xt.device)
    return relpos, Xt1, mask


class RandomShiftsDataSet(Dataset):
    def __init__(self, X, radius, reps):
        self.X = X
        self.shape = self.X.shape[1:]
        self.radius = radius
        self.reps = reps

    def __len__(self):
        return self.reps

    def __getitem__(self, index):
        relpos, X1, mask, = rand_offset_target(self.X, self.radius)
        return self.X, X1, mask, relpos


def make_patches(C, k, pad=False):
    if pad:
        C = np.stack([np.pad(C[i], k//2, 'constant') for i in range(C.shape[0])])
    padded = C.transpose(1, 2, 0)
    patches = extract_patches_2d(padded, (k, k))
    patches = patches.transpose(0, 3, 1, 2)
    return patches


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    method = args.method
    sim = args.sim
    output_dir = f"{args.output}/{args.task}/embeddings/{method}/{sim:03d}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # utils.set_seed(args.seed)
    pl.seed_everything(args.seed)
    dir = f"{args.sims_dir}"

    # simulation data and metadata
    dtrain = dict(np.load(join(dir, args.task, f"{sim:03d}_train.npz")))
    dtest = dict(np.load(join(dir, args.task, f"{sim:03d}_test.npz")))
    with open(join(dir, args.task, "args.yaml"), "r") as io:
        cfg = yaml.safe_load(io)
        ksize = cfg["ksize"]
        radius = ksize // 2

    nd, nr, nc = dtrain["covars"].shape

    # patchify
    C = dtrain['covars']
    Ctest = dtest['covars']

    patches = make_patches(C, ksize, pad=False)
    patches_test = make_patches(Ctest, ksize, pad=False) 
    patches_padded = make_patches(C, ksize, pad=True)

    # train PCA and t-SNE
    if method == "pca":
        X = patches.reshape((patches.shape[0], -1))
        X_padded = patches_padded.reshape((patches_padded.shape[0], -1))
        Xtest = patches_test.reshape((patches.shape[0], -1))
        fitted = PCA(n_components=args.d, svd_solver='arpack').fit(X)
        Z_pca = fitted.transform(X_padded).reshape(nr, nc, -1).transpose(2, 0, 1)
        Z_pca = (Z_pca - Z_pca.mean((1, 2), keepdims=True)) / Z_pca.std((1, 2), keepdims=True)
        np.save(f"{output_dir}/embs.npy", Z_pca)

        Xhat = fitted.inverse_transform(fitted.transform(X))
        Xtest_hat = fitted.inverse_transform(fitted.transform(Xtest))

        R2 = 1.0 - np.sum((X - Xhat)**2) / np.sum((X - X.mean())**2)
        R2_test = 1.0 - np.sum((Xtest - Xtest_hat)**2) / np.sum((Xtest - Xtest.mean())**2)
        results = dict(R2=float(R2), R2_test=float(R2_test), method=method, sim=sim)
        with open(f"{output_dir}/metrics.yaml", "w") as io:
            yaml.dump(results, io)
        logging.info(results)
        return

    elif method == "tsne":
        X = patches.reshape((patches.shape[0], -1))
        X_padded = patches_padded.reshape((patches_padded.shape[0], -1))
        Xtest = patches_test.reshape((patches_test.shape[0], -1))
        fitted = TSNE(n_components=2).fit(X)
        Z_tsne = fitted.transform(X_padded).reshape(nr, nc, -1).transpose(2, 0, 1)
        Z_tsne = (Z_tsne - Z_tsne.mean((1, 2), keepdims=True)) / Z_tsne.std((1, 2), keepdims=True)
        np.save(f"{output_dir}/embs.npy", Z_tsne)
        return

    else:
        input_shape = list(patches.shape)
        kwargs = vars(args)
        if method == "crae":
            model = baselines.CAE(input_shape, depth=2, dh=8, dlat=args.d, vae=False, **kwargs)
        elif method == "cvae":
            model = baselines.CAE(input_shape, depth=2, dh=8, dlat=args.d, vae=True, **kwargs)
        elif method == "unet":
            model = baselines.UNetSelfLearner(nd, depth=2, dh=8, dlat=args.d, **kwargs)
        elif method == "resnet":
            model = baselines.ResNetSelfLearner(nd, dh=8, depth=7, k=3, dlat=args.d, **kwargs)

        if method in ("crae", "cvae"):
            dl_train = TensorDataset(torch.FloatTensor(patches))
            dl_val = TensorDataset(torch.FloatTensor(patches_test))
        else:
            dl_train = RandomShiftsDataSet(torch.FloatTensor(C), radius, reps=nr * nc // 8)
            dl_val = RandomShiftsDataSet(torch.FloatTensor(Ctest), radius, reps=100)

        dloader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        dl_train = DataLoader(dl_train, **dloader_kwargs)
        dl_val = DataLoader(dl_val, **dloader_kwargs)

        torch.use_deterministic_algorithms(True, warn_only=True)
        logsdir = f"{args.output}/{args.task}/embeddings/{method}"
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            enable_progress_bar=args.verbose,
            max_epochs=args.epochs,
            logger=CSVLogger(logsdir, name=f"{sim:003d}", version=""),
            auto_lr_find=args.auto_lr_find
        )
        trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)

        # save embeddings
        if method in ("crae", "cvae"):
            with torch.no_grad():
                Z = model(torch.FloatTensor(patches_padded))
                Z = Z.numpy().reshape(nr, nc, -1).transpose(2, 0, 1)
        else:
            with torch.no_grad():
                Z = model(torch.FloatTensor(C)[None]).numpy()[0]
        Z = (Z - Z.mean((1, 2), keepdims=True)) / Z.std((1, 2), keepdims=True)
        np.save(f"{output_dir}/embs.npy", Z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--sim", type=int, default=0)
    parser.add_argument("--output", type=str, default="results-sim")
    parser.add_argument("--task", type=str, default="nonlinear")
    parser.add_argument("--sims_dir", type=str, default="simulations")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    avail_methods = ["unet", "tsne", "pca", "crae", "cvae", "resnet"]
    parser.add_argument("--method", type=str, default="crae", choices=avail_methods)
    parser.add_argument("--silent", default=True, dest="verbose", action="store_false")
    parser.add_argument("--manual_lr", default=True, dest="auto_lr_find", action="store_false")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    if args.method in ("unet", "resnet"):
        args.batch_size = max(1, args.batch_size // 8)

    main(args)