import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn.modules import batchnorm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models import UNetEncoder, Decoder
from torch.optim.lr_scheduler import LinearLR
import matplotlib.pyplot as plt
import proplot as pplt
import argparse
import logging
import yaml
from utils import load_training_data, nbrs_avg
import pandas as pd
import torchvision.transforms.functional as vF
import argparse
import utils


class NARRData(Dataset):
    def __init__(self, path: str, radius: int, nbr_av_size: int = 0, yearly_average: bool = False):
        super().__init__()
        assert radius >= 0
        self.nbr_av_size = nbr_av_size
        self.radius = radius

        # load data
        C, _, *_ = load_training_data(path, standardize_weather=True, year_averages=yearly_average)
        self.C = torch.FloatTensor(C)
        self.Ctgts = self.C.clone()

        # compute neighbor averages
        if nbr_av_size > 0:
            self.C = nbrs_avg(self.C, nbr_av_size)

    def __len__(self):
        return self.C.shape[0]

    def rand_offset_target(self, tgt, ord=2):
        radius = self.radius
        if radius == 0:
            offset = [0, 0]
        else:
            while True:
                offset = np.random.randint(-radius, radius + 1, size=2)
                if np.linalg.norm(offset, ord=2) <= radius:
                    offset = list(offset)
                    break
        kws = dict(translate=offset, angle=0.0, scale=1.0, shear=0.0)
        shifted_tgt = vF.affine(tgt, **kws)
        mask = vF.affine(torch.ones_like(tgt), fill=0.0, **kws)
        offset = torch.FloatTensor(offset)
        return offset, shifted_tgt, mask

    def __getitem__(self, index):
        Ct = self.C[index]
        offset, Ct1, Mt = self.rand_offset_target(self.Ctgts[index])
        return Ct, Ct1, offset, Mt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhidden", type=int, default=32)
    parser.add_argument("--depthwise", action="store_true", default=False)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--nres", type=int, default=0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--nbrs_av", type=int, default=0)
    parser.add_argument("--yearly_average", default=False, action="store_true")
    parser.add_argument("--datapath", type=str, default="data/training_data.pkl")
    parser.add_argument("--odir", type=str, default="test")
    parser.add_argument("--device", default=0)
    parser.add_argument("--nworkers", type=int, default=0)
    parser.add_argument("--radius", default=5, type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--bntype", default="frn", choices=["bn", "frn"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr0", type=float, default=3e-2)
    parser.add_argument("--lr1", type=float, default=5e-5)
    parser.add_argument("--wdecay", type=int, default=0.0)
    args = parser.parse_args()

    if not args.odir.startswith("results/unsupervised-narr/"):
        args.odir = os.path.join("results/unsupervised-narr", args.odir)
    os.makedirs(args.odir, exist_ok=True)
    # load metadata
    with open(os.path.join(args.odir, "args.yaml"), "w") as io:
        cfg = yaml.dump(vars(args), io)
    logger = utils.make_logger(file=f"{args.odir}/train.log", print=True)
    utils.set_seed(args.seed)

    dataset = NARRData(args.datapath, args.radius, args.nbrs_av, args.yearly_average)
    loader = DataLoader(
        dataset, args.batchsize, True, num_workers=args.nworkers, pin_memory=True
    )
    nt, nd, nr, nc = dataset.C.shape
    dev = args.device if torch.cuda.is_available() else "cpu"

    mkw = dict(
        n_hidden=args.nhidden,
        depth=args.depth,
        num_res=args.nres,
        ksize=args.ksize,
        groups=args.groups,
        batchnorm=True,
        batchnorm_type=args.bntype,
    )
    dkw = dict(batchnorm=True, offset=True, batchnorm_type=args.bntype)
    if not args.local and args.nbrs_av == 0:
        enc = UNetEncoder(nd, args.nhidden, **mkw).to(dev)
        dec = Decoder(args.nhidden, nd, args.nhidden, **dkw).to(dev)
    else:
        enc = nn.Identity()
        dec = Decoder(nd, nd, args.nhidden, **dkw).to(dev)

    mod = nn.ModuleDict(dict(enc=enc, dec=dec))

    logger.debug("Num params")
    for key, m in mod.items():
        logger.debug(f"{key}: {utils.count_parameters(m)}")

    opt = optim.Adam(mod.parameters(), lr=args.lr0, weight_decay=args.wdecay)
    sched = LinearLR(opt, args.lr0, args.lr1, args.epochs)

    # losses = []
    best_lost = np.inf
    for e in range(args.epochs):
        epoch_losses = []
        for sample in tqdm(loader):
            C, C1, offsets, M = [x.to(dev) for x in sample]
            mod.zero_grad()
            Z = mod["enc"](C)
            C1hat = mod["dec"](Z, offsets)
            # SELF-SUPERVISED LOSS (EQ 4 IN THE PAPER)
            loss_t = (M * (C1hat - C1)).pow(2).sum((1, 2, 3))
            loss = loss_t.mean()
            loss.backward()
            ss_t = (M * C1).pow(2).sum((1, 2, 3))
            opt.step()
            for (o1, o2), l, s in zip(offsets, loss_t, ss_t):
                row = dict(ep=e, dr=int(o1), dc=int(o2), sse=l.item(), ss=s.item())
                epoch_losses.append(row)
        df = pd.DataFrame(epoch_losses)
        # losses.extend(df)
        dfmean = df.sum(0)
        metrics = dict(loss=dfmean.sse, r2=1 - dfmean.sse / dfmean.ss)
        logger.info(f"ep: {e}/{args.epochs} | " + utils.format_metrics(metrics))
        sched.step()

        if dfmean.sse < best_lost:
            best_lost = dfmean.sse
            # save model weights 
            torch.save(mod.state_dict(), os.path.join(args.odir, "model.pt"))
    
    # losses = pd.cat(losses)
    # losses.to_csv(f"{args.odir}/losses.csv", index=False)
