from copy import deepcopy
import itertools
import numpy as np
import torch
from torch import nn, optim
import argparse

from torch.optim.lr_scheduler import LinearLR
import models as mods
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os.path import join
from os import X_OK, environ
from collections import defaultdict, deque
import yaml
import wandb
import utils
from sghmc import SG_HMC, SG_HMC_Adam
import torchvision.transforms.functional as vF
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.special import expit
import deprecated.potentials as potentials
from patsy import dmatrix


class CustomDataset(Dataset):
    def __init__(self, D: dict, transform: bool = True, batch_size: int = 8, radi: list[int] = 0, sparse: bool = False):
        self.Y = torch.FloatTensor(D["outcome"])
        self.A = torch.FloatTensor(D["assignment"])
        self.C = torch.FloatTensor(D["covars"])

        self.PS = torch.FloatTensor(D["prob"])
        self.transform = transform
        self.radi = radi
        if transform:
            affine = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                    transforms.RandomAffine(
                    # degrees=10,
                    degrees=0,
                    translate=[0.125, 0.125],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                # transforms.RandomResizedCrop(self.Y.shape, scale=(0.75, 1)),
            ]
            self.affine = transforms.Compose(affine)
        self.batch_size = batch_size
        self.sparse = sparse
        if sparse:  # generate sparse mask
            if transform:
                raise Exception("Sparse augmentation not implemented")
            nr, nc = self.Y.shape
            sparse_mask = np.zeros(nr * nc, dtype=np.float32)
            sparse_mask[np.random.choice(nr * nc, 500, replace=False)] = 1.0
            sparse_mask = sparse_mask.reshape(nr, nc)
            self.sM = torch.FloatTensor(sparse_mask)

        else:
            self.sM = torch.tensor(1.0)



    def rand_offset_target(self, Cbt, Mbt):
        max_rad = np.random.choice(self.radi) + 1
        sgn = np.random.choice([-1, 1], replace=True, size=2)
        relpos = np.random.randint(0, max_rad + 1, size=2)
        relpos *= sgn
        shifted_tgt = vF.affine(
            Cbt, translate=list(relpos), angle=0.0, scale=1.0, shear=0.0
        )
        shifted_mask = vF.affine(
            Mbt[None], translate=list(relpos), angle=0.0, scale=1.0, shear=0.0
        ).squeeze(0)
        relpos = torch.FloatTensor(relpos)
        return relpos, shifted_tgt, shifted_mask

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        nr, nc = self.Y.shape
        rows = (torch.arange(nr * nc) // nc).view(nr, nc)
        cols = (torch.arange(nr * nc) % nc).view(nr, nc)
        RC = torch.stack([rows, cols], -1).to(self.Y.device)
        Y = self.Y.unsqueeze(0)
        A = self.A.unsqueeze(0)
        PS = self.PS.unsqueeze(0)
        M = torch.ones_like(Y)
        X = torch.cat([Y, A, PS, M, self.C])
        if self.transform:
            X = self.affine(X)
        Y = X[0]
        A = X[1]
        PS = X[2]
        M = X[3]
        C = X[4:]
        relpos, C1, M1 = self.rand_offset_target(C, M)
        return Y, A, PS, M, C, RC, relpos, C1, M1

def agg_values(d: dict, key: str, fun: callable = np.mean):
    return {f"{key}_{k}": fun(v) for k, v in d.items()}

def auto_corrcoef(x):
    x = np.array(x)
    return np.corrcoef(x[1:-1], x[2:])[0,1]

def augment(X: torch.Tensor, splines: bool = True, interactions: bool = True):
    C = X
    nt, nd, nr, nc = X.shape
    if splines:
        Cjs = []
        C = C.detach().cpu().numpy()
        for j in range(C.shape[1]):
            Cj = dmatrix("bs(x, 3) - 1", {"x": C[:,j].flatten()})
            Cjs.append(Cj.reshape(nt, nr, nc, 3).transpose(0, 3, 1, 2))
        C = np.stack(Cjs, axis=1)
        C = torch.FloatTensor(np.concatenate(Cjs, axis=1))
    if interactions:
        combs = []
        for i, j in itertools.combinations(range(X.shape[1]), 2):
            combs.append(X[i] * X[j])
        combs = torch.stack(combs, axis=1)
        C = torch.cat([C, combs], axis=1)
    return C

def main(args: argparse.Namespace):
    """Model-based causal inference using U-net"""
    sim = args.sim
    # config mlogger
    mlogger = utils.make_logger(
        file=f"{args.output}/{sim:03d}.log", print=True
    )
    utils.set_seed(args.seed)

    dir = args.dir
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = "cpu"
    ls = 0.00001 # label smoothing
    dout = args.dropout

    # load metadata
    with open(join(dir, "args.yaml"), "r") as io:
        cfg = yaml.safe_load(io)

    nd, nr, nc = 2, 128, 256

    # load data
    dtrain = dict(np.load(join(dir, f"{sim:03d}_train.npz")))
    dtest = dict(np.load(join(dir, f"{sim:03d}_test.npz")))
    if args.old_format:
        covars = np.load(join(dir, "covars_orig.npy"))
        nd = covars.shape[1]  # over-write
        dtrain["covars"] = covars[0]
        dtest["covars"] = covars[0]
        for k in ("prob", "assignment", "outcome"):
            dtrain[k] = dtrain[k][0]
            dtest[k] = dtest[k][0]
    bs = args.batchsize if args.augment else 1
    radi = args.radi if args.radi is not None else cfg["ksize"] // 2
    dset = CustomDataset(
        dtrain,
        batch_size=bs,
        radi=radi,
        transform=args.augment,
        sparse=args.sparse,
    )
    dloader = DataLoader(
        dset,
        batch_size=bs,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    Ytest = dtest["outcome"]
    Atest = dtest["assignment"]
    Ctest = dtest["covars"]
    PStest = dtest["prob"]
    sM = dset.sM.to(dev)
    tau = float(dtrain["effect_size"])

    # make modelsss
    # lt = "mse" if args.mode == "prog" else "binary"
    # lt = "mse"
    lt = "binary"

    kw_local = dict(
        nr=nr,
        nc=nc,
        nd=nd,
        loss_type=lt,
        local=False,
        spatial=args.spatial,
        output_decay=0.0
    )
    kw_spatial = dict(
        nr=nr,
        nc=nc,
        nd=0,
        loss_type=lt,
        local=False,
        spatial=True,
        output_decay=0.0
    )
    kw_avs = dict(
        nr=nr,
        nc=nc,
        nd=2 * nd,
        loss_type=lt,
        local=False,
        spatial=args.spatial,
        output_decay=0.0
    )
    kw_w2vec = kw_local.copy(),
    kw_w2vec_spa = kw_w2vec.copy()
    kw_w2vec_spa["spatial"] = True

    mkw_local = dict(
        n_hidden=args.scale_local,
        depth=0,
        num_res=args.num_res_local,
        ksize=1,
        dropout=dout,
        batchnorm=args.batchnorm
    )
    mkw_avs = mkw_local.copy()
    mkw_w2vec = dict(
        n_hidden=args.scale,
        depth=args.depth,
        num_res=args.num_res,
        ksize=args.ksize,
        groups=args.groups,
        dropout=dout,
        # bottleneck=args.bottleneck,
        batchnorm=args.batchnorm,
        depthwise=args.depthwise,
    )
    mkw_w2vec_unsup = dict(
        n_hidden=args.num_unsup,
        depth=args.depth,
        num_res=args.num_res,
        ksize=args.ksize,
        groups=1,  # args.groups,
        dropout=dout,
        bottleneck=args.bottleneck,
        batchnorm=args.batchnorm,
        depthwise=args.depthwise,
    )

    kw_w2vec_unsup_pscore = kw_w2vec.copy()
    kw_w2vec_unsup_pscore["nd"] = args.num_unsup

    # mA_nocov = mods.SpatialReg(mkw=mkw_nocov, **kw)
    mA_spatial = mods.SpatialReg(**kw_spatial,)

    nrs = 0 if args.sparse else 1
    nh = 16 if args.sparse else 16

    
    din = args.num_unsup
    din += args.interactions * (din * (din - 1) // 2) + args.splines * 3 * din
    unsup_pscore = mods.Decoder(din, 1, nh, nrs, loss_type="binary")

    din = nd + args.interactions * (nd * (nd - 1) // 2) + args.splines * 3 * nd
    local_pscore = mods.Decoder(din, 1, nh, nrs, loss_type="binary")
    avs_pscore = mods.Decoder(2 * din, 1, nh, nrs, loss_type="binary")


    mA_unsup = nn.ModuleDict(dict(
        encoder=mods.UNetEncoder(nd, args.num_unsup, **mkw_w2vec_unsup),
        decoder=mods.Decoder(args.num_unsup, nd, n_hidden=16, bottleneck=args.bottleneck, batchnorm=args.batchnorm, offset=True),
        pscore=unsup_pscore,
    ))
   
    mA_local = local_pscore
    mA_avs = avs_pscore
    mA_w2vec = mods.SpatialReg(mkw=mkw_w2vec, **kw_w2vec)
    mA_w2vec_spa = mods.SpatialReg(mkw=mkw_w2vec, **kw_w2vec_spa)

    modsA = nn.ModuleDict(
        dict(
            spatial=mA_spatial,
            unsup=mA_unsup,
            local=mA_local,
            avs=mA_avs,
            w2vec=mA_w2vec,
            w2vec_spa=mA_w2vec_spa,
        )
    ).to(dev)

    # print number of parameters
    for k, m in modsA.items():
        if not isinstance(m, nn.ModuleDict):
            m = nn.ModuleDict({"model": m})
        for kk, mm in m.items():
            npars = sum(p.numel() for p in mm.parameters() if p.requires_grad)
            mlogger.debug(f"Model {k}-{kk}: num. params={npars}")

    all_pars = {k: m.parameters() for k, m in modsA.items()}
    # opt = optim.SGD(all_pars, momentum=0.5, lr=0.001)
    # opt = SG_HMC(all_pars, lr=lr, temp=(1/nr/nc)**2, weight_decay=1e-3)
    opts = {k: optim.Adam(p, lr=args.lr0, weight_decay=args.wdecay) for k, p in all_pars.items()}
    # schedulers = {k: LinearLR(o, args.lr0, args.lr1, args.epochs) for k, o in opts.items()}
    # temp = (1 / nr / nc) ** 2
    # temp = 1e-6
    temp = 1.0
    # opts = {k: SG_HMC(p, lr=lr, weight_decay=1e-6, temp=temp) for k, p in all_pars.items()}
    # opts = {k: SG_HMC_Adam(p, lr=lr, weight_decay=1e-8, temp=temp, alpha=1.0) for k, p in all_pars.items()}
    # opt = optim.Adam(mA_nocov.parameters(), lr=1e-2, weight_decay=1e-6)

    # move to tensors
    Ytt = torch.FloatTensor(Ytest).to(dev).unsqueeze(0)
    Ctt = torch.FloatTensor(Ctest).to(dev).unsqueeze(0)
    Att = torch.FloatTensor(Atest).to(dev).float().unsqueeze(0)
    PStt = torch.FloatTensor(PStest).to(dev).unsqueeze(0)


    ehat = defaultdict(lambda: 0.0)
    ehat_post = defaultdict(lambda: deque([], maxlen=args.buffer_size))

    needs_grad = []

    for e in range(args.epochs):
        # for opt in opts.values():
        #     opt.optimizing() if e < (args.epochs // 5) else opt.sampling()
        #     for g in opt.param_groups:
        #         g['lr'] = lr if e < (args.epochs // 5) else lr / 1000
        # if e % 2 == 0:  # every 2nd epoch
        #     for par in needs_grad:
        #         par.requires_grad = False
        #     # empty list
        #     needs_grad.clear()

        phase1_error = defaultdict(list)
        phase1_error_test = defaultdict(list)
        outcome_error = defaultdict(list)
        outcome_error_test = defaultdict(list)
        corrs = defaultdict(list)

        sample = next(iter(dloader))
        Yt, At, PSt, Mt, Ct, RC, relpos, C1t1, M1t = [x.to(dev) for x in sample]
        Cnbrst = utils.nbrs_avg(Ct, cfg["ksize"], 1)
        prop_score_losses = {}
        phase1_preds = []

        for k, m in modsA.items():
            opts[k].zero_grad()
            if args.mode == "prop":
                tgt = At.clip(ls, 1.0 - ls)
                if k in ("unsup",):
                    Xt = Ct
                    latent = m['encoder'](Xt)
                    C1hatt = m['decoder'](latent, relpos)
                    MN = Mt * M1t
                    err = C1hatt - C1t1
                    loss_ = (MN * err ** 2).sum()

                    Zt = latent.detach()
                    Zt = augment(Zt, args.splines, args.interactions)
                    L = m['pscore'](Zt)
                  
                    loss_pscore = F.binary_cross_entropy_with_logits(L, tgt, reduction='none')
                    #  torch.where(
                    #     L >= 0.0,
                    #     torch.log(1.0 + torch.exp(-L) + 1e-6) + (1.0 - tgt) * L,
                    #     torch.log(1.0 + torch.exp(L) + 1e-6) - tgt * L,
                    # )
                    loss_pscore = (loss_pscore * MN * sM).sum()
                    loss_ = loss_ + loss_pscore
                    mu_A = At.mean(dim=(1,2), keepdims=True)
                    # mu_err = err.mean(dim=(2,3), keepdims=True)
                    cov = MN * err * (At - mu_A).unsqueeze(1)
                    cov = cov.mean(dim=(2, 3))
                    sig_A = At.std(dim=(1, 2)).unsqueeze(1)
                    sig_err = err.std(dim=(2, 3))
                    # corr = cov / (1e-6 + sig_A * sig_err)
                    # corr_loss = (cov ** 2).mean()
                    # corr_loss = (corr ** 2).mean()

                elif k == "spatial":
                    Mt = Mt * sM
                    Xt = None
                    Utt = None
                    L, loss_ = m.loss(tgt, M=Mt)
                # elif k == "spatial":
                #     Xt = None
                #     Utt = None
                #     Mt = Mt * sM
                #     # L, loss_ = m.loss(tgt, M=Mt)
                #     tmp = RC.view(-1, 2).float()
                #     L = m(tmp)
                #     L = L.view(1, nr, nc)
                #     loss_ = F.binary_cross_entropy_with_logits(L, tgt, reduction='none')
                #     loss_ = (loss_ * Mt).sum() / Mt.sum()
                elif k == "avs":
                    Mt = Mt * sM
                    Xt = torch.cat([Ct, Cnbrst], 1)
                    # Xt = Cnbrst
                    Cnbrst_test = utils.nbrs_avg(Ctt, cfg["ksize"], 1)
                    Xt = augment(Xt, args.splines, args.interactions)
                    Utt = torch.cat([Ctt, Cnbrst_test], 1)
                    Utt = augment(Utt, args.splines, args.interactions)
                    L, loss_ = m.loss(tgt, Xt, M=Mt)
                else:
                    Mt = Mt * sM
                    Xt = Ct
                    Xt = augment(Xt, args.splines, args.interactions)
                    Utt = augment(Ctt, args.splines, args.interactions)
                    L, loss_ = m.loss(tgt, Xt, M=Mt)
                

                # L = logits_.detach()
                # with torch.no_grad():
                #     L = torch.log(PSt / (1.0 - PSt))
                phase1_preds.append(L)
                prop_score_losses[k] = loss_
                phase1_error[k].append(float(loss_))
                with torch.no_grad():
                    m.eval()  # deactivate dropout
                    tgtt = Att
                    # tgtt = torch.log(PStt / (1.0 - PStt))
                    if k not in ("spatial", "unsup", ):
                        _, loss_test = m.loss(tgtt, Utt)
                        loss_test = (loss_test * sM).sum() 
                        phase1_error_test[k].append(float(loss_test))
                    m.train()

        total_loss = sum(prop_score_losses.values())
        for k in modsA.keys():
            prop_score_losses[k].backward()
            # for m in modsA.values():
            #     torch.nn.utils.clip_grad_norm_(all_pars[k], 100.0)
            opts[k].step()
            # schedulers[k].step()

        keys = [("nocov", None)] + list(modsA.items())
        preds = [None] + phase1_preds
        for (k, m), L in zip(keys, preds):
            with torch.no_grad():
                Mt = Mt * sM
                if args.proc == "iptw":
                    # stabilized IPTW estimation
                    N = Mt.sum()
                    pstab = (At * Mt).sum() / N   
                    if k == "nocov":
                        wts = torch.ones_like(Yt)
                    else:
                        # p = L.detach().clip(-4.0, 4.0).sigmoid()
                        p = L.detach().sigmoid()
                        # p = L.sigmoid()
                        # p = PSt
                        wts = torch.where(At.bool(), pstab / (p + 1e-6), (1.0 - pstab) / (1 - p + 1e-6))
                        # wts = torch.where(At.bool(), 1.0 / p, 1.0 / (1.0 - p))
                    Y1 = (wts * Yt * At * Mt).sum() / (wts * At * Mt).sum()
                    Y0 = (wts * Yt * (1.0 - At) * Mt).sum() / (wts * (1 - At) * Mt).sum()
                    effect_pred = (Y1 - Y0).float()
                elif args.proc == "strat":
                    if k == "nocov":
                        Y1 = (Yt * At * Mt).sum() / (At * Mt).sum()
                        Y0 = (Yt * (1.0 - At) * Mt).sum() / ((1 - At) * Mt).sum()
                        effect_pred = (Y1 - Y0).float()
                    else:
                        p = L.detach().clip(-4.0, 4.0).sigmoid()
                        vals = np.zeros((args.nstrata, 2))
                        cnts = np.zeros((args.nstrata, 2))
                        a = At.view(-1).detach().cpu().numpy().astype(int)
                        y = Yt.view(-1).detach().cpu().numpy()
                        m = Mt.view(-1).detach().cpu().numpy().astype(bool)
                        p = p.view(-1).detach().cpu().numpy()
                        for i in range(a.shape[0]):
                            if not m[i]:
                                continue
                            lev = int(p[i] *args.nstrata)
                            vals[lev, a[i]] += float(y[i])
                            cnts[lev, a[i]] += 1
                        mus = vals / (cnts + 1e-6)
                        diffs = mus[:, 1] - mus[:, 0]
                        wts = np.minimum(cnts[:, 0], cnts[:, 1])
                        effect_pred = np.average(diffs, weights=wts/wts.sum())
            key = "tau_" + k

            if not np.isnan(float(effect_pred)):
                current = ehat[key]
                ehat[key] += 0.002 * (float(effect_pred) - current)
            else:
                mlogger.warning("NaN prediction for %s" % key)

            if e % args.thinning == 0:
                if not np.isnan(float(effect_pred)):
                    ehat_post[key].append(float(effect_pred))

        # if (e + 1) % 2000 == 0:  # lower the learning rate
        #     for g in opt.param_groups:
        #         g["lr"] *= 0.1
        #         lr = g["lr"]

        msgdict = {
            **agg_values(phase1_error, "ph1_error"),
            **agg_values(phase1_error_test, "ph1_error_test"),
            **ehat,
            **agg_values(outcome_error, "outcome_error"),
            **agg_values(corrs, "corr"),
            **agg_values(outcome_error_test, "outcome_error_test"),
            **agg_values(ehat_post, "pmean"),
            **agg_values(ehat_post, "pq95", fun=lambda x: np.quantile(x, 0.95)),
            **agg_values(ehat_post, "pq05", fun=lambda x: np.quantile(x, 0.05)),
            **agg_values(ehat_post, "pacorr", fun=auto_corrcoef),
            "train_loss": float(total_loss),
        }
        wandb.log(msgdict)
        if e % 10 == 0:
            mlogger.debug(f"\t===== Epoch {e} ====")
            metrics = [f"{k}: {v:.4f}" for k, v in msgdict.items()]
            mlogger.info(metrics)
            # mlogger.info(msgdict.items())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sim", type=int, default=0)
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--nstrata", type=int, default=10)
    parser.add_argument("--radi", type=list, default=None)
    parser.add_argument("--lr0", type=float, default=1e-4)
    # parser.add_argument("--lr1", type=float, default=5e-5)
    parser.add_argument("--wdecay", type=float, default=1e-3)
    parser.add_argument("--scale_local", type=int, default=16)
    parser.add_argument("--num_unsup", type=int, default=16)
    parser.add_argument("--num_res", type=int, default=0)
    parser.add_argument("--num_res_local", type=int, default=2)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    mc = ["prog", "prop"]  # mode choices
    parser.add_argument("--mode", type=str, default="prop", choices=mc)
    pc = ["iptw", "strat"]  # mode choices
    parser.add_argument("--proc", type=str, default="iptw", choices=pc)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--augment", default=False, action="store_true")
    parser.add_argument("--no_batchnorm", default=False, dest="batchnorm", action="store_false")
    parser.add_argument("--bntype", type="bn", choices=["bn", "frn"])
    parser.add_argument("--dropout", default=False, action="store_true")
    parser.add_argument("--sparse", default=False, action="store_tr ue")
    parser.add_argument("--bottleneck", default=False, action="store_true")
    parser.add_argument("--dir", type=str, default="simulations/test")
    parser.add_argument("--output", type=str, default="simlogs")
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--spatial", default=False, action="store_true")
    parser.add_argument("--splines", default=False, action="store_true")
    parser.add_argument("--interactions", default=False, action="store_true")
    parser.add_argument("--depthwise", default=False, action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--old_format", default=False, action="store_true")
    args = parser.parse_args()

    if not args.wandb:
        environ["WANDB_MODE"] = "offline"
    wandb.init(project="weather2vec-simstudy", config=vars(args))

    np.set_printoptions(precision=4)
    main(args)