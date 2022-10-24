from collections import deque

import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import precision_recall

from models import make_batchnorm
import models


class CAE(pl.LightningModule):
    def __init__(self, input_shape, dh=16, dlat=16, depth=3, k=3, lr=0.0003, factor=2, vae=False, patience=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.vae = vae
        self.patience = patience
        self.SE, self.SEv, self.SS, self.SSv = 0.0, 0.0, 0.0, 0.0
        self.Smeans = deque(maxlen=500)
        self.Svmeans = deque(maxlen=500)
        self.input_shape = input_shape
        self.encoded_shape = np.array(input_shape[-2:]) // (2**depth)
        self.encoded_dim = dh * 2**depth
        self.latent_size = self.encoded_dim * np.prod(self.encoded_shape)
        q = factor
        din = input_shape[1]

        def down_block(din_, dout_, k_, pool=True, act=True, bn=True):
            return nn.Sequential(
                nn.Conv2d(din_, dout_, k_, padding='same'),
                nn.SiLU() if act else nn.Identity(),
                make_batchnorm(dout_, 'frn') if bn else nn.Identity(),
                nn.AvgPool2d(2, 2, 0) if pool else nn.Identity(),
            )
        
        def up_block(din_, dout_, k_, act=True, bn=True):
            return nn.Sequential(
                nn.ConvTranspose2d(din_, dout_, k_, stride=2),
                nn.SiLU() if act else nn.Identity(),
                make_batchnorm(dout_, 'frn') if bn else nn.Identity(),
            )

        self.down_net = nn.Sequential(
            down_block(din, dh, k, pool=False),
            *[down_block(dh * q**i, dh * q**(i + 1), k) for i in range(depth)]
        )

        if self.vae:
            self.fc = nn.Linear(self.latent_size, 2 * dlat)
        else:
            self.fc = nn.Linear(self.latent_size, dlat)
        self.fc_transpose = nn.Sequential(
            nn.SiLU(), nn.Linear(dlat, self.latent_size), nn.SiLU(),
        )
        self.up_net = nn.Sequential(
            *[up_block(dh * q**(i + 1), dh * q**i, k) for i in reversed(range(depth))],
            down_block(dh, din, k, act=False, bn=False, pool=False),
        )

    def reparameterize(self, mu, logvar, sample=True):
        sig = torch.exp(torch.clip(0.5 * logvar, -10.0, 5.0)) * float(sample)
        return mu + sig * torch.randn_like(mu)

    def forward(self, inputs, sample=True):
        Z = self.down_net(inputs).reshape(inputs.shape[0], -1)
        if self.vae:
            mu, logvar = torch.chunk(self.fc(Z), 2, axis=1)
            return self.reparameterize(mu, logvar, sample)
        else:
            return self.fc(Z)

    def unpad(self, Shat):
        h, w = self.input_shape[2:]
        h0 = (Shat.shape[2] - h) // 2
        w0 = (Shat.shape[3] - w) // 2
        return Shat[:,:,h0:(h0 + h), w0:(w0 + w)]

    def training_step(self, batch, _):
        S = batch[0]
        Z = self.down_net(S).view(S.shape[0], -1)
        if self.vae:
            mu, logvar = torch.chunk(self.fc(Z), 2, axis=1)
            Z = self.reparameterize(mu, logvar, sample=True)
        else:
            Z = self.fc(Z)
        Z = self.fc_transpose(Z)
        Z = Z.view(Z.shape[0], self.encoded_dim, *self.encoded_shape)
        Shat = self.up_net(Z)
        Shat = self.unpad(Shat)

        # vi loss
        if self.vae:
            kld_loss = -0.5 * torch.sum(1.0 + logvar - mu**2 - logvar.exp(), dim = 1).mean(0)
        else:
            kld_loss = torch.tensor(0.0).to(S.device)
        mse_loss = F.mse_loss(S, Shat)
        loss = kld_loss + mse_loss

        self.Smeans.append(S.mean().item())
        self.SE += ((S - Shat).pow(2).mean() - self.SE).item()
        self.SS += ((S - np.mean(self.Smeans)).pow(2).mean() - self.SS).item()
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('kld_loss', kld_loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('mse_loss', mse_loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        S = batch[0]
        Z = self.down_net(S).view(S.shape[0], -1)
        if self.vae:
            mu, logvar = torch.chunk(self.fc(Z), 2, axis=1)
            Z = self.reparameterize(mu, logvar, sample=False)
        else:
            Z = self.fc(Z)
        Z = self.fc_transpose(Z)
        Z = Z.view(Z.shape[0], self.encoded_dim, *self.encoded_shape)
        Shat = self.up_net(Z)
        Shat = self.unpad(Shat)

        mse_loss = F.mse_loss(S, Shat)
        self.Svmeans.append(S.mean().item())
        self.SEv += ((S - Shat).pow(2).mean() - self.SEv).item()
        self.SSv += ((S - np.mean(self.Svmeans)).pow(2).mean() - self.SSv).item()
        self.log('vloss', mse_loss.item(), on_epoch=True, on_step=False, prog_bar=True)

    def on_validation_epoch_start(self):
        self.SEv = 0.0
        self.SSv = 0.0

    def on_validation_epoch_end(self):
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def on_train_epoch_start(self):
        self.SS = 0.0
        self.SS = 0.0

    def on_train_epoch_end(self):
        r2 = 1.0 - self.SE / self.SS
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt


class UNetSelfLearner(pl.LightningModule):
    def __init__(self, din, dh=2, dlat=2, depth=3, k=3, lr=0.0003, patience=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.patience = patience
        self.lr = lr
        self.SE, self.SEv, self.SS, self.SSv = 0.0, 0.0, 0.0, 0.0

        self.enc = models.UNetEncoder(
            cin=din,
            n_hidden=dh,
            cout=dlat,
            depth=depth,
            ksize=k,
            num_res=0,
            pool='AvgPool2d',
            batchnorm_type="frn",
            batchnorm=True
        )
        self.predec = nn.Sequential(nn.SiLU(), make_batchnorm(dlat, 'frn'))
        self.dec = models.Decoder(dlat, din, n_hidden=16, offset=True)

    def forward(self, inputs):
        return self.enc(inputs)
    
    def training_step(self, batch, batch_idx):
        S, S1, mask, offset = batch
        S1hat = self.dec(self.predec(self(S)), offset)
        S = S * mask
        S1 = S1 * mask
        mse_loss = F.mse_loss(S1, S1hat)
        S1mean = S1[np.where(mask.detach().cpu().numpy())].mean()
        self.SE += ((S1 - S1hat).pow(2).mean() - self.SE).item()
        self.SS += ((S1 - S1mean).pow(2).mean() - self.SS).item()
        r2 = 1.0 - self.SE/self.SS
        self.log('eloss', mse_loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)
        return mse_loss
    
    def validation_step(self, batch, batch_idx):
        S, S1, mask, offset = batch
        S1hat = self.dec(self.predec(self(S)), offset)

        S = S * mask
        S1 = S1 * mask
        mse_loss = F.mse_loss(S1, S1hat)
        S1mean = S1[np.where(mask.detach().cpu().numpy())].mean()
        self.SEv += ((S1 - S1hat).pow(2).mean() - self.SEv).item()
        self.SSv += ((S1 - S1mean).pow(2).mean() - self.SSv).item()
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vloss', mse_loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # opt = torch.optim.SGD(self.parameters(), momentum=0.9, lr=self.lr)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt


class ResNetSelfLearner(pl.LightningModule):
    def __init__(self, din, dh=2, dlat=2, depth=3, k=3, lr=0.0003, patience=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.patience = patience
        self.lr = lr
        self.SE, self.SEv, self.SS, self.SSv = 0.0, 0.0, 0.0, 0.0

        def make_block(din_, dout_, k_, act=True,):
            return nn.Sequential(
                nn.Conv2d(din_, dout_, k_, padding='same'),
                nn.SiLU() if act else nn.Identity(),
            )
        self.first = nn.Sequential(
            make_block(din, dh, k),
            make_batchnorm(dh, 'frn')
        )
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        for _ in range(depth - 1):
            self.convs.append(make_block(dh, dh, k))
            self.acts.append(make_batchnorm(dh, 'frn'))

        self.final = make_block(dh, dlat, k, act=False)
        self.predec = nn.Sequential(nn.SiLU(), make_batchnorm(dlat, 'frn'))
        self.dec = models.Decoder(dlat, din, n_hidden=16, offset=True)

    def forward(self, inputs):
        x = self.first(inputs)
        for f, a in zip(self.convs, self.acts):
            x = a(x + f(x))
        return self.final(x)
    
    def training_step(self, batch, _):
        S, S1, mask, offset = batch
        S1hat = self.dec(self.predec(self(S)), offset)
        S = S * mask
        S1 = S1 * mask
        mse_loss = F.mse_loss(S1, S1hat)
        S1mean = S1[np.where(mask.detach().cpu().numpy())].mean()
        self.SE += ((S1 - S1hat).pow(2).mean() - self.SE).item()
        self.SS += ((S1 - S1mean).pow(2).mean() - self.SS).item()
        r2 = 1.0 - self.SE/self.SS
        self.log('eloss', mse_loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)
        return mse_loss
    
    def validation_step(self, batch, _):
        S, S1, mask, offset = batch
        S1hat = self.dec(self.predec(self(S)), offset)

        S = S * mask
        S1 = S1 * mask
        mse_loss = F.mse_loss(S1, S1hat)
        S1mean = S1[np.where(mask.detach().cpu().numpy())].mean()
        self.SEv += ((S1 - S1hat).pow(2).mean() - self.SEv).item()
        self.SSv += ((S1 - S1mean).pow(2).mean() - self.SSv).item()
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vloss', mse_loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt


class FFNGridClassifier(pl.LightningModule):
    def __init__(self, din, dh, lr=0.001, depth=3, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.lr = lr
        self.wd = weight_decay
        self.patience = patience
        def make_block(din_, dout_, act=True):
            return nn.Sequential(
                nn.Conv2d(din_, dout_, 1),
                nn.SiLU() if act else nn.Identity(),
                make_batchnorm(dout_, 'frn') if act else nn.Identity()
            )
        self.net = nn.Sequential(
            make_block(din, dh if depth > 0 else 1, act=depth > 0),
            *[make_block(dh, dh) for _ in range(depth - 1)],
            make_block(dh, 1, act=False) if depth > 0 else nn.Identity()
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)
    
    def training_step(self, batch, _):
        Z, A, M = batch
        logits = self.net(Z).squeeze(1)
        ix = np.where(M.cpu().numpy())
        loss = F.binary_cross_entropy_with_logits(logits[ix], A[ix])
        with torch.no_grad():
            Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
            prec, recall = precision_recall(Ahat, A.long())
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('prec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('recall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        Z, A , M = batch
        logits = self.net(Z).squeeze(1)
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A)
        Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
        prec, recall = precision_recall(Ahat, A.long())
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vprec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vrecall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt


class WXClassifier(pl.LightningModule):
    def __init__(self, din, k, lr=0.001, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.wd = weight_decay
        self.patience = patience
        self.lr = lr
        self.k = k if isinstance(k, (tuple, list)) else (k, k)
        self.net = nn.Conv2d(din, 1, self.k, padding='same')

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)
    
    def training_step(self, batch, _):
        Z, A, M = batch
        logits = self.net(Z).squeeze(1)
        # g = self.k//2
        # M = M[...,g:-g, g:-g].cpu().numpy()
        # logits, A = logits[...,g:-g, g:-g], A[...,g:-g, g:-g]
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A)
        with torch.no_grad():
            Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
            prec, recall = precision_recall(Ahat, A.long())
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('prec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('recall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        Z, A, M = batch
        logits = self.net(Z).squeeze(1)
        # g = self.k//2
        # M = M[...,g:-g, g:-g].cpu().numpy()
        # logits, A = logits[...,g:-g, g:-g], A[...,g:-g, g:-g]
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A)
        Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
        prec, recall = precision_recall(Ahat, A.long())
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vprec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vrecall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt



class UNetClassifier(pl.LightningModule):
    def __init__(self, din, dh=2, depth=3, k=3, lr=0.0003, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.patience = patience
        self.k = k
        self.wd = weight_decay
        self.lr = lr
        self.net = models.UNetEncoder(
            cin=din,
            n_hidden=dh,
            cout=1,
            depth=depth,
            ksize=k,
            num_res=0,
            batchnorm_type="frn",
            batchnorm=True
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)
    
    def training_step(self, batch, _):
        Z, A, M = batch
        logits = self.net(Z).squeeze(1)
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A)
        with torch.no_grad():
            Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
            prec, recall = precision_recall(Ahat, A.long())
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('prec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('recall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        Z, A, M = batch
        logits = self.net(Z).squeeze(1)
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A)
        Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
        prec, recall = precision_recall(Ahat, A.long())
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vprec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vrecall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt


class ResNetClassifier(pl.LightningModule):
    def __init__(self, din, k, dh, depth=3, lr=0.001, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.patience = patience
        self.lr = lr
        self.wd = weight_decay
        self.k = k

        def make_block(din_, dout_, k_, act=True):
            return nn.Sequential(
                nn.Conv2d(din_, dout_, k_, padding='same'),
                nn.SiLU() if act else nn.Identity(),
            )
        self.first = nn.Sequential(
            make_block(din, dh, k),
            make_batchnorm(dh, 'frn')
        )
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        for _ in range(depth - 1):
            self.convs.append(make_block(dh, dh, k))
            self.acts.append(make_batchnorm(dh, 'frn'))
        self.final = make_block(dh, 1, k, act=False)

    def forward(self, inputs):
        x = self.first(inputs)
        for f, a in zip(self.convs, self.acts):
            x = a(x + f(x))
        return self.final(x).squeeze(1)
    
    def training_step(self, batch, _):
        Z, A, M = batch
        logits = self(Z)
        ix = np.where(M.cpu().numpy())
        loss = F.binary_cross_entropy_with_logits(logits, A)
        logits, A = logits[ix], A[ix]
        with torch.no_grad():
            Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
            prec, recall = precision_recall(Ahat, A.long())
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('prec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('recall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        Z, A, M = batch
        logits = self(Z).squeeze(1)
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A)
        Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
        prec, recall = precision_recall(Ahat, A.long())
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vprec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vrecall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt


class CARClassifier(pl.LightningModule):
    def __init__(self, nr, nc, lr=0.001, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.patience = patience
        self.wd = weight_decay
        self.lr = lr
        self.net = models.GMRF(nr, nc)

    def forward(self, inputs=None):
        return self.net()
    
    def training_step(self, batch, _):
        Z, A, M = batch
        logits = self()
        rloss = self.net.penalty()
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A) + 10 * rloss
        with torch.no_grad():
            Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
            prec, recall = precision_recall(Ahat, A.long())
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('rloss', rloss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('prec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('recall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        Z, A, M = batch
        logits = self()
        rloss = self.net.penalty()
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A) + rloss
        Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
        prec, recall = precision_recall(Ahat, A.long())
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('rloss', rloss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vprec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vrecall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt


class UNetCARClassifier(pl.LightningModule):
    def __init__(self, nr, nc, din, dh=2, depth=3, k=3, lr=0.0003, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.patience = patience
        self.k = k
        self.lr = lr
        self.wd = weight_decay
        self.net = models.UNetEncoder(
            cin=din,
            n_hidden=dh,
            cout=1,
            depth=depth,
            ksize=k,
            num_res=0,
            pool='AvgPool2d',
            batchnorm_type="frn"
        )
        self.car = models.GMRF(nr, nc, intercept=False)

    def forward(self, inputs):
        return self.net(inputs).squeeze(1) + self.car()
    
    def training_step(self, batch, _):
        Z, A, M = batch
        logits = self.net(Z).squeeze(1) + self.car()
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        rloss = self.car.penalty()
        loss = F.binary_cross_entropy_with_logits(logits, A) + rloss
        with torch.no_grad():
            Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
            prec, recall = precision_recall(Ahat, A.long())
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('rloss', rloss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('prec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('recall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        Z, A, M = batch
        logits = self.net(Z).squeeze(1) + self.car()
        rloss = self.car.penalty()
        ix = np.where(M.cpu().numpy())
        logits, A = logits[ix], A[ix]
        loss = F.binary_cross_entropy_with_logits(logits, A) + rloss
        Ahat = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
        prec, recall = precision_recall(Ahat, A.long())
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('rloss', rloss.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vprec', prec.item(), on_epoch=True, on_step=False, prog_bar=True)
        self.log('vrecall', recall.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        if self.patience > 0:
            sched = ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-5, patience=self.patience, verbose=True, threshold=2e-3)
            return [opt], [dict(scheduler=sched, monitor='eloss')]  
        else:
            return opt


class WXRegression(pl.LightningModule):
    def __init__(self, din, k, lr=0.001, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.patience = patience
        self.wd = weight_decay
        self.lr = lr
        self.k = k if isinstance(k, (tuple, list)) else (k, k)
        self.net = nn.Conv2d(din, 1, self.k, bias=True, padding='same')

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)
    
    def training_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SS += (Y - Y.mean()).pow(2).sum().item()
        self.SE += (Y - Yhat).pow(2).sum().item()
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SSv += (Y - Y.mean()).pow(2).sum().item()
        self.SEv += (Y - Yhat).pow(2).sum().item()
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.SEv = 0.0
        self.SSv = 0.0

    def on_train_epoch_start(self):
        self.SE = 0.0
        self.SS = 0.0

    def on_validation_epoch_end(self):
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def on_train_epoch_end(self):
        r2 = 1.0 - self.SE / self.SS
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        # opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt


class UNetRegression(pl.LightningModule):
    def __init__(self, din, dh=2, depth=3, k=3, lr=0.0003, patience=0, weight_decay=0.0, factor=2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.patience = patience
        self.wd = weight_decay
        self.lr = lr
        self.net = models.UNetEncoder(
            cin=din,
            n_hidden=dh,
            cout=1,
            depth=depth,
            ksize=k,
            num_res=0,
            batchnorm_type="frn",
            batchnorm=True,
            factor=factor,
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)
    
    def training_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SS += (Y - Y.mean()).pow(2).sum().item()
        self.SE += (Y - Yhat).pow(2).sum().item()
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SSv += (Y - Y.mean()).pow(2).sum().item()
        self.SEv += (Y - Yhat).pow(2).sum().item()
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.SEv = 0.0
        self.SSv = 0.0

    def on_train_epoch_start(self):
        self.SE = 0.0
        self.SS = 0.0

    def on_validation_epoch_end(self):
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def on_train_epoch_end(self):
        r2 = 1.0 - self.SE / self.SS
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt


class CARUNetRegression(pl.LightningModule):
    def __init__(self, nr, nc, din, dh=2, depth=3, k=3, lr=0.0003, lam=1.0, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.patience = patience
        self.wd = weight_decay
        self.lr = lr
        self.lam = lam
        self.net = models.UNetEncoder(
            cin=din,
            n_hidden=dh,
            cout=1,
            depth=depth,
            ksize=k,
            num_res=0,
            batchnorm_type="frn",
            batchnorm=True
        )
        self.car = models.GMRF(nr, nc, fit_scale=(lam > 0.0), intercept=False)

    def forward(self, inputs):
        return self.net(inputs).squeeze(1) + self.car()
    
    def training_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M) + self.lam * self.car.penalty()
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SS += (Y - Y.mean()).pow(2).sum().item()
        self.SE += (Y - Yhat).pow(2).sum().item()
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M) + self.lam * self.car.penalty()
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SSv += (Y - Y.mean()).pow(2).sum().item()
        self.SEv += (Y - Yhat).pow(2).sum().item()
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.SEv = 0.0
        self.SSv = 0.0

    def on_train_epoch_start(self):
        self.SE = 0.0
        self.SS = 0.0

    def on_validation_epoch_end(self):
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def on_train_epoch_end(self):
        r2 = 1.0 - self.SE / self.SS
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        # opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt


class FFNGridRegression(pl.LightningModule):
    def __init__(self, din, dh, lr=0.001, depth=3, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.lr = lr
        self.wd = weight_decay
        self.patience = patience
        def make_block(din_, dout_, act=True):
            return nn.Sequential(
                nn.Conv2d(din_, dout_, 1),
                nn.SiLU() if act else nn.Identity(),
                make_batchnorm(dout_, 'frn') if act else nn.Identity()
            )
        self.net = nn.Sequential(
            make_block(din, dh if depth > 0 else 1, act=depth > 0),
            *[make_block(dh, dh) for _ in range(depth - 1)],
            make_block(dh, 1, act=False) if depth > 0 else nn.Identity()
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)
    
    def training_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SS += (Y - Y.mean()).pow(2).sum().item()
        self.SE += (Y - Yhat).pow(2).sum().item()
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SSv += (Y - Y.mean()).pow(2).sum().item()
        self.SEv += (Y - Yhat).pow(2).sum().item()
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.SEv = 0.0
        self.SSv = 0.0

    def on_train_epoch_start(self):
        self.SE = 0.0
        self.SS = 0.0

    def on_validation_epoch_end(self):
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def on_train_epoch_end(self):
        r2 = 1.0 - self.SE / self.SS
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt


class NSUNetRegression(pl.LightningModule):
    def __init__(self, nr, nc, din, dh=2, dlat=2, depth=3, k=3, lr=0.0003, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.patience = patience
        self.wd = weight_decay
        self.lr = lr
        self.Ymean = 0.0
        self.net = models.UNetEncoder(
            cin=din,
            n_hidden=dh,
            cout=dlat,
            depth=depth,
            ksize=k,
            num_res=0,
            batchnorm_type="frn",
            batchnorm=True,
            final_act=True,
        )
        self.beta = nn.Parameter(torch.zeros(dlat, nr, nc), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(nr, nc), requires_grad=True)

    def forward(self, inputs):
        return (self.net(inputs) * self.beta).sum(1) + self.alpha
    
    def training_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        # replicate the r2 metric of Shen et al. (2017)
        # r2shen = compute_r2shen(Y, Yhat, M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.Ymean = Y.mean()
        self.SS += (Y - self.Ymean).pow(2).sum().item()
        self.SE += (Y - Yhat).pow(2).sum().item()
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        # self.log('r2shen', r2shen.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SSv += (Y - self.Ymean).pow(2).sum().item()
        self.SEv += (Y - Yhat).pow(2).sum().item()
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.SEv = 0.0
        self.SSv = 0.0

    def on_train_epoch_start(self):
        self.SE = 0.0
        self.SS = 0.0

    def on_validation_epoch_end(self):
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def on_train_epoch_end(self):
        r2 = 1.0 - self.SE / self.SS
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt




class NSWXRegression(pl.LightningModule):
    def __init__(self, nr, nc, din, dlat=2, k=3, lr=0.0003, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.patience = patience
        self.wd = weight_decay
        self.lr = lr
        self.Ymean = 0.0
        self.net = nn.Conv2d(din, dlat, k, padding='same')
        self.beta = nn.Parameter(torch.zeros(dlat, nr, nc), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(nr, nc), requires_grad=True)

    def forward(self, inputs):
        return (self.net(inputs) * self.beta).sum(1) + self.alpha
    
    def training_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        M = M.cpu().numpy()
        ix = np.where(M)

        # replicate the r2 metric of Shen et al. (2017)
        # r2shen = compute_r2shen(Y, Yhat, M)
        Y, Yhat = Y[ix], Yhat[ix]
        self.Ymean = Y.mean()
        self.SS += (Y - self.Ymean).pow(2).sum().item()
        self.SE += (Y - Yhat).pow(2).sum().item()
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        # self.log('r2shen', r2shen, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SSv += (Y - self.Ymean).pow(2).sum().item()
        self.SEv += (Y - Yhat).pow(2).sum().item()
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.SEv = 0.0
        self.SSv = 0.0

    def on_train_epoch_start(self):
        self.SE = 0.0
        self.SS = 0.0

    def on_validation_epoch_end(self):
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def on_train_epoch_end(self):
        r2 = 1.0 - self.SE / self.SS
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt



class NSLocalRegression(pl.LightningModule):
    def __init__(self, nr, nc, din, lr=0.0003, patience=0, weight_decay=0.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.din = din
        self.patience = patience
        self.wd = weight_decay
        self.lr = lr
        self.Ymean = 0.0
        self.beta = nn.Parameter(torch.zeros(din, nr, nc), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(nr, nc), requires_grad=True)

    def forward(self, inputs):
        return (inputs * self.beta).sum(1) + self.alpha
    
    def training_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        M = M.cpu().numpy()
        ix = np.where(M)

        # replicate the r2 metric of Shen et al. (2017)
        # r2shen = compute_r2shen(Y, Yhat, M)
        Y, Yhat = Y[ix], Yhat[ix]
        self.Ymean = Y.mean()
        self.SS += (Y - self.Ymean).pow(2).sum().item()
        self.SE += (Y - Yhat).pow(2).sum().item()
        self.log('eloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        # self.log('r2shen', r2shen, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        C, Y, M = batch
        Yhat = self(C)
        loss = F.mse_loss(Yhat * M, Y * M)
        ix = np.where(M.cpu().numpy())
        Y, Yhat = Y[ix], Yhat[ix]
        self.SSv += (Y - self.Ymean).pow(2).sum().item()
        self.SEv += (Y - Yhat).pow(2).sum().item()
        self.log('vloss', loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.SEv = 0.0
        self.SSv = 0.0

    def on_train_epoch_start(self):
        self.SE = 0.0
        self.SS = 0.0

    def on_validation_epoch_end(self):
        vr2 = 1.0 - self.SEv / self.SSv
        self.log('vr2', vr2, on_epoch=True, on_step=False, prog_bar=True)

    def on_train_epoch_end(self):
        r2 = 1.0 - self.SE / self.SS
        self.log('r2', r2, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.wd)
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt

