from collections import defaultdict
from copy import deepcopy
from email.policy import default
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from unet import UNetEncoder, Decoder, MeanOnlyDecoder
from torch.optim.lr_scheduler import LinearLR
import matplotlib.pyplot as plt

import proplot as pplt
import argparse
import logging
import yaml
from utils import nbrs_avg
import pandas as pd
import torchvision.transforms.functional as vF
import argparse
from patsy import dmatrix
from utils import load_training_data
import utils


class NARRData(Dataset):
    def __init__(self, path: str, nbr_av_size: int = 0, idxs: list = None, seasonal: bool = False):
        super().__init__()
        self.nbr_av_size = nbr_av_size
        self.seasonal = seasonal

        # load data
        C, _, Y, M = load_training_data(
            path, standardize_weather=True, log_so4=True, standardize_so4=True, year_averages=True, remove_zeros=False
        )
        *_, self.nr, self.nc = C.shape
        self.C = torch.FloatTensor(C)
        self.Y = torch.FloatTensor(Y)
        self.M = torch.FloatTensor(M).repeat((Y.shape[0], 1, 1))
        if idxs is not None:
            self.C = self.C[range(idxs[0], idxs[1])]
            self.Y = self.Y[range(idxs[0], idxs[1])]
            self.M = self.M[range(idxs[0], idxs[1])]

        # compute neighbor averages
        if nbr_av_size > 0:
            self.C = torch.cat([self.C, nbrs_avg(self.C, nbr_av_size)], 1)

    def __len__(self):
        return self.C.shape[0]

    def __getitem__(self, index):
        Ct = self.C[index]
        Yt = self.Y[index]
        Mt = self.M[index] 
        if self.seasonal:
            St = torch.ones_like(Yt) * (index % 12)
            Ct = torch.cat([Ct, St.unsqueeze(0)])
        return Ct, Yt, Mt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhidden", type=int, default=8)
    parser.add_argument("--nhidden-dec", type=int, default=128)
    parser.add_argument("--nres-dec", type=int, default=2)
    parser.add_argument("--depthwise", action="store_true", default=False)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--unadjusted", action="store_true", default=False)
    parser.add_argument("--nres", type=int, default=0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--av-nbrs", type=int, default=0)
    parser.add_argument("--datapath", type=str, default="data/training_data.pkl")
    parser.add_argument("--odir", type=str, default="test")
    parser.add_argument("--device", default=0)
    parser.add_argument("--nworkers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--lr0", type=float, default=3e-2)
    parser.add_argument("--lr1", type=float, default=1e-4)
    parser.add_argument("--wdecay", type=float, default=1e-3)
    parser.add_argument("--bntype", default="frn", choices=["bn", "frn"])
    parser.add_argument("--test-range", default=[0, 12], type=int)
    parser.add_argument("--train-range", default=[12, 72], type=int)
    parser.add_argument("--seasonal", action="store_true", default=False)
    args = parser.parse_args()

    if not args.odir.startswith("results/supervised-so4/"):
        args.odir = os.path.join("results/supervised-so4", args.odir)
    os.makedirs(args.odir, exist_ok=True)
    # load metadata
    with open(os.path.join(args.odir, "args.yaml"), "w") as io:
        cfg = yaml.dump(vars(args), io)
    logger = utils.make_logger(file=f"{args.odir}/train.log", print=True)
    utils.set_seed(args.seed)

    train_dataset = NARRData(args.datapath, args.av_nbrs, idxs=args.train_range, seasonal=args.seasonal)
    test_dataset = NARRData(args.datapath, args.av_nbrs, idxs=args.test_range, seasonal=args.seasonal)
    full_dataset = NARRData(args.datapath, args.av_nbrs, seasonal=args.seasonal)
    dws = dict(num_workers=args.nworkers, pin_memory=True)
    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True, **dws)
    test_loader = DataLoader(test_dataset, args.batchsize, shuffle=True, **dws)
    full_loader = DataLoader(full_dataset, 1, shuffle=False, **dws)
    _, nd, *_ = train_dataset.C.shape
    if args.seasonal:
        nd += 1
    dev = args.device if torch.cuda.is_available() else "cpu"

    mkw = dict(
        n_hidden=args.nhidden,
        depth=args.depth,
        num_res=args.nres,
        ksize=args.ksize,
        groups=args.groups,
        batchnorm=True,
        depthwise=args.depthwise,
        batchnorm_type=args.bntype
    )
    dkw = dict(batchnorm=True, offset=False, batchnorm_type=args.bntype)
    if args.unadjusted:
        enc = nn.Identity()
        dec = MeanOnlyDecoder().to(dev)
    elif args.local or args.av_nbrs > 0:
        enc = nn.Identity()
        dec = Decoder(nd, 1, args.nhidden_dec, n_res=args.nres_dec, **dkw).to(dev)
    else:
        enc = UNetEncoder(nd, args.nhidden, **mkw).to(dev)
        dec = Decoder(args.nhidden, 1, args.nhidden_dec, n_res=0, **dkw).to(dev)

    mod = nn.ModuleDict(dict(enc=enc, dec=dec))

    logger.debug("Num params")
    for key, m in mod.items():
        logger.debug(f"{key}: {utils.count_parameters(m)}")

    opt = optim.Adam(mod.parameters(), lr=args.lr0, weight_decay=args.wdecay)
    # opt = optim.SGD(mod.parameters(), momentum=0.5, lr=args.lr0, weight_decay=args.wdecay)
    sched = LinearLR(opt, args.lr0, args.lr1, args.epochs)

    best_lost = np.inf
    for e in range(args.epochs):
        epoch_train_losses = []
        mod.train()
        for sample in tqdm(train_loader):
            C, Y, M = [x.to(dev) for x in sample]
            mod.zero_grad()
            Z = mod["enc"](C)
            Yhat = mod["dec"](Z).squeeze(1)
            # SUPERVISED PROGNOSTIC LOSS (EQ 3 IN THE PAPER)
            loss_t = (M * (Yhat - Y)).pow(2).sum((1, 2))
            loss = loss_t.mean()
            loss.backward()
            ss = (M * Y).pow(2).sum((1, 2))
            opt.step()
            row = dict(sse=loss.item(), ss=ss.mean().item())
            epoch_train_losses.append(row)
        df = pd.DataFrame(epoch_train_losses).mean(0)
        metrics = dict(train_loss=df.sse, train_r2=1 - df.sse / df.ss)

        mod.eval()
        epoch_test_losses = []
        for sample in tqdm(test_loader):
            C, Y, M = [x.to(dev) for x in sample]
            Z = mod["enc"](C)
            Yhat = mod["dec"](Z).squeeze(1)
            loss_t = (M * (Yhat - Y)).pow(2).sum((1, 2))
            loss = loss_t.mean()
            ss_t = (M * Y).pow(2).sum((1, 2))
            row = dict(sse=loss.item(), ss=ss_t.mean().item())
            epoch_test_losses.append(row)
        df = pd.DataFrame(epoch_test_losses).mean(0)
        metrics.update(dict(test_loss=df.sse, test_r2=1 - df.sse / df.ss))

        logger.info(f"ep: {e}/{args.epochs} | " + utils.format_metrics(metrics))
        sched.step()

        if df.sse < best_lost:
            best_lost = df.sse
            torch.save(mod.state_dict(), os.path.join(args.odir, "model.pt"))

