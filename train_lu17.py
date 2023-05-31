import argparse
from collections import defaultdict
import os
import yaml
import logging

import numpy as np
from scipy import stats
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import baselines



def compute_r2shen(Y, Yhat, M, plim=1.0):
    n, nr, nc = Y.shape
    corrmat = torch.full((nr, nc), torch.nan)
    ninvalid = 0
    for i in range(nr):
        for j in range(nc):
            with torch.no_grad():
                valid = M[:, i,j] > 0
                n = float(sum(valid))
                if n <= 2: continue
                mat = torch.stack([Y[valid, i, j], Yhat[valid, i, j]], 0)
                corr = torch.corrcoef(mat)[0, 1]
                t0 = stats.t.ppf(1. - 0.5 * plim, df=n - 1)
                rlim = np.sqrt(t0**2 / (n-2 + t0**2))
                if corr.abs() >= rlim:
                    corrmat[i, j] = corr
                else:
                    ninvalid += 1
    return torch.nanmean(corrmat**2), ninvalid



def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    method = args.method
    output_dir = f"{args.output}/{method}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # utils.set_seed(args.seed)
    pl.seed_everything(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # read datase
    data = np.load("data_simstudy/lu2017.npz")
    
    # Y = np.log(1 + data['monthly_PM']).transpose(2, 1, 0)
    Y = data['monthly_PM_std'].transpose(2, 1, 0)
    # Y = data['monthly_PM_std'].transpose(2, 1, 0)
    C = data['covars'].transpose(3, 0, 2, 1)  # ntxndxnrxnc
    Y = (Y - np.nanmean(Y)) / np.nanstd(Y)
    C = (C - np.nanmean(C, (0, 2, 3), keepdims=True)) / np.nanstd(C, (0, 2, 3), keepdims=True)

    # make masks
    # M = (~np.isnan(Y) & ~np.any(np.isnan(C), 1)).astype(np.float32)
    M = (~np.isnan(Y)).astype(np.float32)
    Y[np.isnan(Y)] = 0.0
    C[np.isnan(C)] = 0.0
    C, Y, M = [torch.FloatTensor(u) for u in (C, Y, M)]
    nt, nd, nr, nc = C.shape

    if "unet" in method:
        Y, M = Y.unsqueeze(1), M.unsqueeze(1)
        C, Y, M = [F.interpolate(u, [12, 28], mode='bilinear') for u in (C, Y, M)]
        Y, M = Y.squeeze(1), M.squeeze(1)
        nr, nc = 12, 28

    # if "ns_" in method:
    #     row = np.arange(nr * nc).reshape(nr, nc) // nc
    #     col = np.arange(nr * nc).reshape(nr, nc) % nc
    #     nsvar = np.stack([row, col], 0)
    #     nsvar = np.stack(nt * [nsvar], 0)
    #     nsvar = torch.FloatTensor(nsvar)
    #     nsvar = (nsvar - nsvar.mean()) / nsvar.std()
    #     if method != "ns_ffn":
    #         C = torch.cat([C, nsvar], 1)
    #         nd += 2
    #     else:
    #         C = nsvar
    #         nd = 2

    metrics = defaultdict(list)
    ncv = nt
    # cvts = np.random.choice(range(nt), ncv, replace=False)
    cvts = np.arange(nt)
    val_preds = []
    # if method == "local_hre":
    #     model = baselines.NSLocalRegression(nr, nc, nd,  **vars(args))
    # if method == "wx_hre":
    #     model = baselines.NSWXRegression(nr, nc, nd, dlat=4, k=(13, 9), **vars(args))
    # if method == "ns_unet":
    #     model = baselines.NSUNetRegression(nr, nc, nd, dh=5, depth=1,  **vars(args))
    # if method == "unet_car":
    #     model = baselines.CARUNetRegression(nr, nc, nd, dh=5, depth=1,  **vars(args))
    # if method == "unet_hre":
    #     model = baselines.CARUNetRegression(nr, nc, nd, dh=5, depth=1,  lam=0.0, **vars(args))
    # if method == "ns_unet_car":
    #     model = baselines.CARUNetRegression(nr, nc, nd, dh=5, depth=1,  **vars(args))
    # if method == "ns_unet_hre":
    #     model = baselines.CARUNetRegression(nr, nc, nd, dh=5, depth=1, lam=0.0, **vars(args))

    for i_t, t in enumerate(cvts):
        
        if method == "wx":
            model = baselines.WXRegression(nd, (13, 9), **vars(args))
        if method == "ns":
            model = baselines.NSWXRegression(nr, nc, din=0, dlat=0, **vars(args))
        if method == "ns_wx":
            model = baselines.NSWXRegression(nr, nc, nd, dlat=4, k=(13, 9), **vars(args))
        if method == "unet":
            model = baselines.UNetRegression(nd, factor=1, depth=1,  **vars(args))
        if method == "ns_unet":
            # model = baselines.NSUNetRegression(nr, nc, nd, factor=1, depth=1,  dlat=4, **vars(args))
            model = baselines.NSUNetRegression(nr, nc, nd, factor=1, depth=0,  dlat=4, **vars(args))
        if method == "unet_car":
            model = baselines.CARUNetRegression(nr, nc, nd, factor=1, depth=0,  **vars(args))
        if method == "ns_local":
            model = baselines.NSWXRegression(nr, nc, nd, dlat=0, **vars(args))
        if method == "local":
            model = baselines.WXRegression(nd, 1, **vars(args))
        if method == "local_ffn":
            model = baselines.FFNGridRegression(nd, 16, **vars(args))
        # if method == "ns_ffn":
        #     model = baselines.FFNGridRegression(nd, 16, **vars(args))
        if method == "ns_local_ffn":
            model = baselines.FFNGridRegression(nd, 16, **vars(args))
        else:
            pass

        ix = (np.arange(nt) != t)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            enable_progress_bar=args.verbose,
            max_epochs=args.epochs,
            logger=CSVLogger(output_dir, name="", version=""),
            # auto_lr_find=True
        )
        dstrain = TensorDataset(C[ix], Y[ix], M[ix])
        dsval = TensorDataset(C[t, None], Y[t, None], M[t, None])
        dltrain = DataLoader(dstrain, pin_memory=True, batch_size=nt - 1) 
        dlval = DataLoader(dsval, pin_memory=True)
        trainer.fit(model, train_dataloaders=dltrain, val_dataloaders=dlval)
        
        metrics_t = trainer.logged_metrics.copy()
        # add r2shen
        r2shen, _ = compute_r2shen(Y[ix], model(C[ix]), M[ix])
        metrics_t['r2shen'] = r2shen.item()

        for k, v in metrics_t.items():
            metrics[k].append(v)
        
        logging.info(f"i_t: {i_t}")
        logging.info(metrics_t)

        Yval_hat = model(C[t, None])[0]
        val_preds.append(Yval_hat)

        prog = {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in metrics.items()}

        Yval_hat = torch.stack(val_preds)
        r2shen_val, ninvalid = compute_r2shen(Y[cvts[:(i_t + 1)]], Yval_hat + 1e-4 * torch.randn_like(Yval_hat), M[cvts[:(i_t + 1)]])
        
        frac_invalid = ninvalid / M[cvts[:(i_t + 1)]].sum()
        prog.update(dict(r2shen_val=r2shen_val.item(), sample_size=i_t, invalid=frac_invalid.item()))
        logging.info(f"Progress: {prog}")
        with open(f"{output_dir}/metrics.yaml", 'w') as io:
            yaml.dump(prog, io)

    # metrics = {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in metrics.items()}
    
    # Yval_hat = torch.stack(val_preds)
    # r2shen_val, ninvalid = compute_r2shen(Y[cvts], Yval_hat, M[cvts])
    # metrics['r2shen_val'] = r2shen_val.item()
    # metrics['ninvalid'] = ninvalid / M[cvts.sum()]
    # logging.info(metrics)

    # with open(f"{output_dir}/metrics.yaml", 'w') as io:
    #     yaml.dump(metrics, io)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dh", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--output", type=str, default="results-lu17")
    parser.add_argument("--task", type=str, default="nonlinear")
    parser.add_argument("--sims_dir", type=str, default="simulations")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    # avail_methods = ["wx"]
    parser.add_argument("--method", type=str, default="ns_unet")
    parser.add_argument("--silent", default=True, dest="verbose", action="store_false")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    main(args)