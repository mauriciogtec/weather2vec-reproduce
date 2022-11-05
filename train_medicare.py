import argparse
import yaml
import logging
import rasterio
# from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import baselines
from utils import load_medicare_data


def main(args: argparse.ArgumentParser):
    pl.seed_everything(args.seed)
    C, A, Y, M, Mtrain, Mtest = load_medicare_data("data/medicare", args.masknum)
    C, A, Y = [torch.FloatTensor(u[None]) for u in (C, A, Y)]
    nd = C.shape[1]
    Ctrain, Ctest = C, C
    Atrain, Atest = A, A
    Ytrain, Ytest = Y, Y
    Mtrain, Mtest, M = [torch.FloatTensor(u) for u in (Mtrain, Mtest, M)]
    dstrain = TensorDataset(Ctrain, Atrain, Ytrain, Mtrain)
    dsval = TensorDataset(Ctest, Atest, Ytest, Mtest)


    dloader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    dltrain = DataLoader(dstrain, batch_size=args.batch_size, **dloader_kwargs)
    dlval = DataLoader(dsval, batch_size=1, **dloader_kwargs)

    model = baselines.UNetCausalPosRegression(nd, **vars(args))

    logsdir = f"{args.odir}/sim{args.masknum:03d}"
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=args.verbose,
        max_epochs=args.epochs,
        logger=CSVLogger(logsdir, name="", version=args.suffix),
        auto_lr_find=args.auto_lr_find,
        # callbacks=[EarlyStopping(monitor="vrmse", mode="min", patience=50)]
    )
    trainer.fit(model, train_dataloaders=dltrain, val_dataloaders=dlval)
    # causal effect
    model.eval()

    with torch.no_grad():
        Y1 = model(C, torch.ones_like(A))
        Y0 = model(C, torch.zeros_like(A))
        ate = (Y1[(M > 0)] - Y0[(M > 0)]).mean().item()
        ate_rel = ((Y1[(M > 0)] - Y0[(M > 0)]) / (1e-6 + Y0[(M > 0)])).mean().item()
        Yr = model(C, A)
        mean_diff = (Yr[(M > 0) & (A > 0.)].mean() - Yr[(M > 0) & (A == 0.)].mean()).item()
        sig = F.mse_loss(Yr, Y).sqrt()
        Yobs = Yr + sig * torch.randn_like(Yr).cpu().numpy()

    D = dict(
        assignment=A.cpu().numpy(),
        covars=C.cpu().numpy(),
        outcome=Yobs.cpu().numpy()
    )
    np.savez(f"{logsdir}/simdata.npz", **D)

    results = dict(ate=ate, ate_rel=ate_rel, mean_diff=mean_diff, rmse=sig.item())
    logging.info(results)
    with open(f"{logsdir}/metrics.yaml", 'w') as io:
        yaml.dump(results, io)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--odir", type=str, default="results/medicare")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dh", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--fksize", type=int, default=1)
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--masknum", type=int, default=0)
    parser.add_argument("--ffn_depth", type=int, default=3)
    parser.add_argument("--ffn_dh", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--conf_penalty", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    # parser.add_argument("--direct", action="store_true", default=True)
    parser.add_argument("--auto_lr_find", default=False, action="store_true")
    parser.add_argument("--silent", default=True, dest="verbose", action="store_false")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)