import argparse
import os
import yaml
from os.path import join
import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger

import baselines
import utils
    

def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    method = args.method
    sim = args.sim
    args.weight_decay = 1e-4 if args.sparse else 1e-6
    subtask = "effects_sparse" if args.sparse else "effects"
    output_dir = f"{args.embsdir}/{args.task}/{subtask}/{method}/{sim:03d}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # utils.set_seed(args.seed)
    pl.seed_everything(args.seed)
    dir = args.dir

    # simulation data and metadata
    dtrain = dict(np.load(join(dir, args.task, f"{sim:03d}_train.npz")))
    dtest = dict(np.load(join(dir, args.task, f"{sim:03d}_test.npz")))
    with open(join(dir, args.task, "args.yaml"), "r") as io:
        cfg = yaml.safe_load(io)
        ksize = cfg["ksize"]
        radius = ksize // 2

    nd, nr, nc = dtrain["covars"].shape

    # patchify
    names = ['covars', 'assignment', 'outcome']
    C, A, Y = [torch.FloatTensor(dtrain[u]) for u in names]
    Ctest, Atest, Ytest = [torch.FloatTensor(dtest[u]) for u in names]

    # normalize input covariates
    Cmeans, Cstd = C.mean((1, 2), keepdim=True), C.std((1, 2), keepdim=True)
    C, Ctest = [(u - Cmeans) / Cstd for u in (C, Ctest)]

    if method == "wx":
        model = baselines.WXClassifier(nd, ksize, **vars(args))
    if method == "causal_wx":
        model = baselines.CausalWXRegression(nd, ksize, **vars(args))
    elif method == "unet_sup":
        model = baselines.UNetClassifier(nd, depth=1, dh=8, k=3, factor=1, **vars(args))
    elif method == "unet_sup_car":
        model = baselines.UNetCARClassifier(nr, nc, nd, depth=1, dh=4, k=3, **vars(args))
    elif method == "resnet_sup":
        model = baselines.ResNetClassifier(nd, depth=6, dh=12, k=3, **vars(args))
    elif method == "car":
        model = baselines.CARClassifier(nr, nc, **vars(args))
    else:
        if method == "avg":
            Cavg = utils.nbrs_avg(C[None], ksize//2, 1).squeeze(0)
            Ctestavg = utils.nbrs_avg(Ctest[None], ksize//2, 1).squeeze(0)
            C = torch.cat([C, Cavg], 0)
            Ctest = torch.cat([Ctest, Ctestavg], 0)
        elif method in ("unet", "tsne", "pca", "crae", "cvae", "resnet"):
            embfile = f"{args.embsdir}/{args.task}/embeddings/{method}/{sim:03d}/embs.npy"
            C = torch.FloatTensor(np.load(embfile))
            # C = torch.cat([C, A[None]], 0) # debug
            # C = torch.randn_like(C)  # debug
            Ctest = C  # TODO: fix this to save the real test embeddings
            Atest = A
        dh = 64 if args.sparse else 64
        depth = 1 if args.sparse else 2
        model = baselines.FFNGridClassifier(C.shape[0], dh=dh, depth=depth, **vars(args))

    if args.sparse:
        sparse_mask = np.zeros(nr * nc, dtype=np.float32)
        sparse_mask[np.random.choice(nr * nc, 500, replace=False)] = 1.0
        sparse_mask = sparse_mask.reshape(nr, nc)
    else:
        sparse_mask = np.ones((nr, nc), dtype=np.float32)
    M = torch.FloatTensor(sparse_mask)

    if method == "causal_wx":
        dl_train = TensorDataset(C[None], A[None], Y[None], M[None])
        dl_val = TensorDataset(Ctest[None], Atest[None], Ytest[None], M[None])
    else:
        dl_train = TensorDataset(C[None], A[None], M[None])
        dl_val = TensorDataset(Ctest[None], Atest[None], M[None])

    dloader_kwargs = dict(
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    dl_train = DataLoader(dl_train, **dloader_kwargs)
    dl_val = DataLoader(dl_val, **dloader_kwargs)

    torch.use_deterministic_algorithms(True, warn_only=True)
    logsdir = f"{args.embsdir}/{args.task}/{subtask}/{method}"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        enable_progress_bar=args.verbose,
        max_epochs=args.epochs,
        logger=CSVLogger(logsdir, name=f"{sim:003d}", version=""),
        auto_lr_find=args.auto_lr_find
    )
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)
    
    # exit early for causal_wx
    if method == "causal_wx":
        effect_estimate = model.effect.item()
    else:
        # evaluate propensity scores and IPTW estimates
        with torch.no_grad():
            pscore = torch.sigmoid(torch.clip(model(C[None]), -10.0, 10.0)).view(nr, nc)
            W = (A * M).sum() / M.sum() 
            wts = A * (W / pscore) + (1.0 - A) * (1 - W) / (1.0 - pscore)
            wts = wts * M
            Y1 = (wts * Y * A).sum() / (wts * A).sum()
            Y0 = (wts * Y * (1.0 - A)).sum() / (wts * (1 - A)).sum()
        effect_estimate = float((Y1 - Y0).item())
    ate_error = float(effect_estimate - dtrain['effect_size'])
    effects = dict(effect_estimate=effect_estimate, ate_error=ate_error, method=method, sim=sim, task=args.task)
    with open(f"{output_dir}/effect.yaml", "w") as io:
        yaml.dump(effects, io)
    logging.info(effects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--sim", type=int, default=0)
    parser.add_argument("--embsdir", type=str, default="results-sim")
    parser.add_argument("--dir", type=str, default="simulations")
    parser.add_argument("--task", type=str, default="nonlinear")
    parser.add_argument("--num_workers", type=int, default=4)
    avail_methods = ["tsne", "pca", "crae", "cvae", "unet_sup", "unet_sup_car",
                     "resnet_sup", "wx", "causal_wx", "unet", "local", "avg", "car", "resnet"]
    parser.add_argument("--method", type=str, default="unet_sup", choices=avail_methods)
    parser.add_argument("--silent", default=True, dest="verbose", action="store_false")
    parser.add_argument("--sparse", default=False, action="store_true")
    parser.add_argument("--manual_lr", default=True, dest="auto_lr_find", action="store_false")
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    main(args)