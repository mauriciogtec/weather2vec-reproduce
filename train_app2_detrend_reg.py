import argparse
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from utils import load_training_data
import baselines


def main(args: argparse.ArgumentParser):
    method = args.method
    pl.seed_everything(args.seed)

    C, _, Y, M  = load_training_data(
        args.datapath, standardize_weather=True, log_so4=False,
        standardize_so4=True, year_averages=False, remove_zeros=True
    )

    # add time to C (the treatment here)
    nt, nd, nr, nc = C.shape
    if args.add_time:
        time = np.zeros((nt, 2, nr, nc))
        for i in range(nr):
            for j in range(nc):
                time[:, 0, i, j] = np.arange(nt) // 12
                time[:, 1, i, j] = np.arange(nt) % 12
        C = np.concatenate([C, time], axis=1)
        nd += 2

    if args.random_test_split:
        datasplits = train_test_split(C, Y, M, test_size=0.1)
        Ctrain, Ctest, Ytrain, Ytest, Mtrain, Mtest = [torch.FloatTensor(u) for u in datasplits]
    else:
        ixtrain = np.arange(*args.train_range)
        ixtest = np.arange(*args.test_range)
        Ctrain, Ytrain, Mtrain = [torch.FloatTensor(u[ixtrain]) for u in (C, Y, M)]
        Ctest, Ytest, Mtest = [torch.FloatTensor(u[ixtest]) for u in (C, Y, M)]

    dstrain = TensorDataset(Ctrain, Ytrain, Mtrain)
    dsval = TensorDataset(Ctest, Ytest, Mtest)

    dloader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    dltrain = DataLoader(dstrain, batch_size=args.batch_size, **dloader_kwargs)
    dlval = DataLoader(dsval, batch_size=1, **dloader_kwargs)

    if method == "wx":
        model = baselines.WXRegression(nd, (13, 9), **vars(args))
    if method == "ns":
        model = baselines.NSWXRegression(nr, nc, din=0, dlat=0, **vars(args))
    if method == "ns_wx":
        model = baselines.NSWXRegression(nr, nc, nd, dlat=4, k=(13, 9), **vars(args))
    if method == "unet":
        model = baselines.UNetRegression(nd, factor=1, **vars(args))
    if method == "resnet":
        model = baselines.ResNetRegression(nd, **vars(args))
    if method == "ns_unet":
        model = baselines.NSUNetRegression(nr, nc, nd, factor=1, dlat=4, **vars(args))
    if method == "unet_car":
        model = baselines.CARUNetRegression(nr, nc, nd, factor=1, **vars(args))
    if method == "ns_local":
        model = baselines.NSWXRegression(nr, nc, nd, dlat=0, **vars(args))
    if method == "local":
        model = baselines.WXRegression(nd, 1, **vars(args))
    if method == "local_ffn":
        model = baselines.FFNGridRegression(nd, 16, **vars(args))
    if method == "ns_local_ffn":
        model = baselines.FFNGridRegression(nd, 16, **vars(args))
    else:
        pass

    logsdir = f"{args.odir}/time_reg"
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=args.verbose,
        max_epochs=args.epochs,
        logger=CSVLogger(logsdir, name=f"depth{args.depth}", version=args.suffix),
        auto_lr_find=args.auto_lr_find
    )
    trainer.fit(model, train_dataloaders=dltrain, val_dataloaders=dlval)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="data/training_data.pkl")
    parser.add_argument("--odir", type=str, default="results")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dh", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--test-range", default=[0, 12], type=int)
    parser.add_argument("--train-range", default=[12, 72], type=int)
    parser.add_argument("--random_test_split", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="unet")
    parser.add_argument("--manual_lr", default=False, dest="auto_lr_find", action="store_false")
    parser.add_argument("--silent", default=True, dest="verbose", action="store_false")
    parser.add_argument("--no_time", default=True, dest="add_time", action="store_false")

    args = parser.parse_args()
    main(args)