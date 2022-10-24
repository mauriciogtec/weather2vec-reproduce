import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from timm.models.layers import trunc_normal_
import pytorch_lightning as pl
        

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def conv_block(
    din: int,
    dout: int,
    k: int,
    act: nn.Module = nn.Identity,
    depthwise: bool = False,
    bottleneck: bool = False,
    batchnorm: bool = False,
    separable: bool = False,
    batchnorm_type: str = "bn",
    **kwargs
):

    if not depthwise:
        d1 = din // 4 if bottleneck else din
        kwargs = kwargs.copy()
        kwargs["bias"] = True # (not batchnorm)
        batchnorm = True # (not batchnorm)
        mods = []
        mods.append(nn.Conv2d(d1, dout, k, **kwargs))
        if batchnorm:
            mods.append(make_batchnorm(dout, batchnorm_type))
        mods.append(act())
        if bottleneck:
            blayer = [nn.Conv2d(din, din // 4, 1, bias=(not batchnorm))]
            if batchnorm:
                blayer.append(make_batchnorm(din // 4, batchnorm_type))
            mods = mods + blayer
    elif separable:
        kwargs = kwargs.copy()
        kwargs2 = kwargs.copy()
        kwargs3 = kwargs.copy()
        kwargs["groups"] = din
        kwargs["padding"] = [0, k//2]
        kwargs["bias"] = True
        kwargs2["groups"] = din
        kwargs2["padding"] = [k//2, 0]
        kwargs2["bias"] = False
        kwargs3["groups"] = 1
        kwargs3["padding"] = 0
        kwargs3["bias"] = (not batchnorm)
        mods = [
            nn.Conv2d(din, din, (1, k), **kwargs),
            nn.Conv2d(din, din, (k, 1), **kwargs2),
            nn.Conv2d(din, dout, 1, **kwargs3),
        ]
        if batchnorm:
            mods.append(make_batchnorm(dout, batchnorm_type))
        mods.append(act())
    else:
        kwargs = kwargs.copy()
        kwargs2 = kwargs.copy()
        kwargs["groups"] = din
        kwargs["padding"] = "same"
        kwargs["bias"] = True
        kwargs2["groups"] = 1
        kwargs2["padding"] = "same"
        kwargs2["bias"] = True  # 
        mods = [
            nn.Conv2d(din, din, (k, k), **kwargs),
            nn.Conv2d(din, dout, 1, **kwargs2)
        ]
        if batchnorm:
            mods.append(make_batchnorm(dout, batchnorm_type))
        mods.append(act())
    mod = nn.Sequential(*mods)
    return mod


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_eps_leanable=False):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable

        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        x = self.weight * x + self.bias
        return x



class res_layers(nn.Module):
    def __init__(self, d: int, n: int, k: int, act: nn.Module, **kwargs):
        super().__init__()
        kwargs = kwargs.copy()
        self.convs = nn.ModuleList(
            [conv_block(d, d, k, padding="same", batchnorm=True, act=nn.Identity) for _ in range(n)]
        )
        self.acts = nn.ModuleList([act() for _ in range(n)])

    def forward(self, inputs: Tensor):
        x = inputs
        for f, act in zip(self.convs, self.acts):
            x = act(f(x) + x)
        return x


def down_layer(d: int, k: int, act: nn.Module, num_res=0, pool=nn.MaxPool2d, factor: int = 2, **kwargs):
    modules = [
        pool(2, 2, 0), conv_block(d // factor, d, k, act, **kwargs)
    ]
    if num_res > 0:
        modules.append(res_layers(d, num_res, k, act, **kwargs))
    return nn.Sequential(*modules)


class up_layer(nn.Module):
    def __init__(self, d: int, k: int, act: nn.Module, num_res=0, factor: int = 2, **kwargs):
        super().__init__()
        modules = []
        self.factor = factor
        if num_res > 0:
            modules.append(res_layers(factor * d, num_res, k, act, **kwargs))
        modules.append(nn.UpsamplingBilinear2d(scale_factor=2))
        modules.append(conv_block(factor * d, d, k, act, **kwargs))
        self.net1 = nn.Sequential(*modules)
        self.net2 = conv_block(2 * d, d, k, act, **kwargs)

    def forward(self, x: Tensor, z: Tensor):
        x = self.net1(x)
        x = torch.cat([x, z], 1)
        x = self.net2(x)
        return x


def make_batchnorm(dim, type):
    if type == "bn":
        return nn.BatchNorm2d(dim, eps=1e-2)
    elif type == "frn":
        return FRN(dim)
    elif type == "layer":
        return LayerNorm(dim)
    else:
        raise NotImplementedError


class UNetEncoder(nn.Module):
    def __init__(
        self,
        cin,
        cout,
        ksize=3,
        depth=3,
        num_res=1,
        groups: int = 1,
        n_hidden: int = 16,
        factor: int = 2,
        act=nn.SiLU,
        pool='MaxPool2d',
        dropout: bool = False,
        bottleneck: bool = False,
        depthwise: bool = False,
        separable: bool = False,
        batchnorm: bool = False,
        batchnorm_type: str = "bn",
        final_act: bool = False
    ) -> None:
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.num_res = num_res
        self.depth = depth
        self.n_hidden = n_hidden
        self.ksize = ksize
        self.groups = groups
        d, k = n_hidden, ksize
        self.act = act
        self.batchnorm = batchnorm
        act = getattr(nn, act) if isinstance(act, str) else act
        pool = getattr(nn, pool)

        kwargs = dict(
            bias=True,
            groups=groups,
            padding="same",
            padding_mode="replicate",
            bottleneck=bottleneck,
            depthwise=depthwise,
            separable=separable,
            batchnorm=batchnorm,
            batchnorm_type=batchnorm_type,
            factor=factor
        )


        kwargs1 = kwargs.copy()
        kwargs1["groups"] = 1
        self.first = nn.Sequential(
            nn.Conv2d(cin, d, k, padding="same"),
            make_batchnorm(d, batchnorm_type),
            act(),
            res_layers(d, num_res, k, act, **kwargs)
        )

        self.down = nn.ModuleList()
        for _ in range(depth):
            d *= factor
            layer = down_layer(d, k, act, num_res=num_res, pool=pool, **kwargs)
            self.down.append(layer)

        if dropout:
            self.dropout = nn.Dropout2d(0.5)

        self.up = nn.ModuleList()
        for _ in range(depth):
            d //= factor
            layer = up_layer(d, k, act, num_res=num_res, **kwargs)
            self.up.append(layer)

        kwargsf = kwargs.copy()
        kwargsf["groups"] = 1
        self.final = nn.Sequential(
            res_layers(d, num_res, k, act, **kwargs),
            nn.Conv2d(d, cout, k, padding="same"),
            nn.SiLU() if final_act else nn.Identity()
            # conv_block(d, cout, k, **kwargsf)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Tensor):
        x = inputs
        x = self.first(x)

        # downsampling
        intermediate_outputs = [x]
        for _, fdown in enumerate(self.down):
            x = fdown(x)
            intermediate_outputs.append(x)

        # dropout
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        # upsampling
        for i, fup in enumerate(self.up):
            z = intermediate_outputs[-(i + 2)]
            x = fup(x, z)

        # final layers and desired size
        x = self.final(x)

        return x


class Decoder(nn.Module):
    def __init__(self, din, dout, n_hidden=64, n_res=0, act=nn.SiLU, offset=False, collapse_dims=True, loss_type="mse", fit_sigma: bool = False, **kwargs):
        super().__init__()
        if offset:
            self.spatial_emb = SpatialEmbeddings(din, n_hidden, n_hidden)
        self.din = din
        self.dout = dout
        self.collapse_dims =collapse_dims
        self.hin = n_hidden if offset else din
        self.head = nn.Sequential(
            conv_block(self.hin, n_hidden, 1, act, **kwargs),
            *[conv_block(n_hidden, n_hidden, 1, act, **kwargs) for _ in range(n_res)],
            nn.Conv2d(n_hidden, dout, 1),
        )
        self.loss_type = loss_type
        if loss_type == "mse":
            self.log_sig = nn.Parameter(torch.tensor(0.0), requires_grad=fit_sigma)

        # self.head = conv_block(n_hidden, n_hidden, 1, act, **kwargs)

    def forward_offset(self, latent, offset):
        spatial = self.spatial_emb(offset)
        bs, D, D1 = spatial.shape
        *_, nr, nc = latent.shape
        spatial = spatial.view(bs, D, D1, 1, 1)
        latent = latent.view(bs, D, 1, nr, nc)
        v = (latent * spatial).view(bs, D, D1, nr, nc)
        out = v.sum(1) # .view(1, D, nr, nc)
        out = self.head(out)
        return out

    def forward(self, latent, offset=None):
        if offset is not None:
            out = self.forward_offset(latent, offset)
        else:
            out = self.head(latent)
        if self.collapse_dims:
            out = out.squeeze(1)
        return out

    def loss_binary(
        self,
        tgt: Tensor,
        C: Optional[Tensor] = None,
        M: Optional[Tensor] = None,
    ):
        if M is None:
            M = torch.ones_like(tgt)
        L = self(C)
        if len(L.shape) == 0:
            L = torch.ones_like(tgt) * L
        loss = F.binary_cross_entropy_with_logits(L, tgt, reduction='none')
        loss = (loss * M).sum() / tgt.shape[0]
        return L, loss
    
    def loss_mse(
        self,
        Y: Tensor,
        C: Optional[Tensor] = None,
        A: Optional[Tensor] = None,
        M: Optional[Tensor] = None,
    ):
        if M is None:
            M = torch.ones_like(Y)
        Yhat = self.forward(C, A) * M
        # return Yhat, 0.0
        sig = (1e-4 + self.log_sig).exp()
        prec = 1.0 / (1e-4 + sig ** 2)
        err = 0.5 * prec * F.mse_loss(Yhat, Y, reduction="none") + self.log_sig
        prior = sig - self.log_sig
        # err = 0.5 * F.mse_loss(Yhat, Y, reduction="none") * M
        loss = ((err * M).sum() + prior)
        # sigerr = F.smooth_l1_loss(mse.detach().m,sqrt(), sig)
        loss = loss  / Y.shape[0]
        return Yhat, loss

    def loss(self, *args, **kwargs):
        if self.loss_type == "binary":
            l = self.loss_binary(*args, **kwargs)
        elif self.loss_type == "mse":
            l = self.loss_mse(*args, **kwargs)
        return l 


class MeanOnlyDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, latent):
        X = torch.ones_like(latent) * self.alpha
        return X[:, :1]


class SpatialEmbeddings(nn.Module):
    def __init__(self, dout, nl, n_hidden=64, act=nn.SiLU, n_chunks=1):
        super().__init__()
        self.D = dout
        self.nl = nl
        act = getattr(nn, act) if isinstance(act, str) else act
        self.dec = nn.Sequential(
            nn.Linear(2, n_hidden),
            act(),
            nn.Linear(n_hidden, n_hidden),
            act(),
            nn.Linear(n_hidden, dout * nl),
        )

    def forward(self, inputs):
        u = inputs
        u = (
            [u] #+
            # [torch.sin(u / (1000 ** (2 * k / 12))) for k in range(0, 11, 2)]
            # + [torch.cos(u / (1000 ** (2 * k / 12))) for k in range(0, 11, 2)]
            # [torch.sin(2 * np.pi * u / k) for k in range(1, 12, 2)] +
            # [torch.cos(2 * np.pi * u / k) for k in range(1, 12, 2)]
        )
        u = torch.cat(u, -1)
        out = self.dec(u)
        out = out.view(-1, self.D, self.nl)
        return out


class GMRF(nn.Module):
    def __init__(self, nrow: int, ncol: int, fit_scale: bool = True, intercept: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(nrow, ncol))
        self.intercept = intercept
        self.log_var = nn.Parameter(torch.tensor(0.0), requires_grad=fit_scale)

    def forward(self):
        out = torch.sqrt(self.log_var.exp()) * self.W.unsqueeze(0)
        return out

    def penalty(self, alph: float = 1.0, beta: float = 1.0, lam: float = 1.0):
        var = (1e-6 + self.log_var).clip(-8.0, 3.0).exp()
        Z = self.W
        N = np.prod(self.W.shape)
        d1 = (Z[:-1] - Z[1:]).pow(2)
        d2 = (Z[:, :-1] - Z[:, 1:]).pow(2)
        tv_reg = 0.5 * (d1.sum() + d2.sum()) #/ N
        prior_reg = (alph + 1.0) * torch.log(var + 1e-6) + beta / (var + 1e-6)
        if not self.intercept:
            center_reg = lam * F.smooth_l1_loss(Z, torch.zeros_like(Z), reduction="mean")
        else:
            center_reg = 0.0
        return tv_reg + prior_reg + center_reg


class SpatialReg(nn.Module):
    def __init__(
        self,
        nr: int = 1,
        nc: int = 1,
        nd: int = 1,
        local: bool = False,
        spatial: bool = False,
        loss_type: str = "mse",
        causal: bool = False,
        mkw: dict = dict(),
        fit_sigma: bool = False,
        conv_only: bool = False,
        output_decay: float = 0.0,
        **kwargs
    ):
        assert loss_type in ("binary", "mse")
        super().__init__()
        self.ncovars = nd
        self.spatial = spatial
        self.causal = causal
        self.output_decay = output_decay
        if spatial:
            self.sp = GMRF(nr, nc, intercept=(nd == 0))

        if causal:
            self.effect_size = nn.Parameter(torch.tensor(0.0))
        self.loss_type = loss_type

        if loss_type == "mse":
            self.log_sig = nn.Parameter(torch.tensor(0.0), requires_grad=fit_sigma)

        if nd == 0:
            self.net = nn.Parameter(torch.tensor(0.0))
        elif conv_only:
            ksize = mkw.get('ksize', 3)
            self.net = nn.Conv2d(nd, 1, ksize, padding="same")
        elif local:
            self.net = ResLocalMLP(nd, 1, **mkw)
        else:
            self.net = UNetEncoder(nd, 1, **mkw)

    def forward(self, C: Optional[Tensor] = None, A: Optional[Tensor] = None):
        out = self.sp() if self.spatial else 0.0
        if C is not None:
            out = out + self.net(C).squeeze(1)
        else:
            out = out + self.net
        if A is not None and self.causal:
            out = out + A * self.effect_size
        return out

    def loss_binary(
        self,
        tgt: Tensor,
        C: Optional[Tensor] = None,
        M: Optional[Tensor] = None,
):
        if M is None:
            M = torch.ones_like(tgt)
        L = self(C)
        if len(L.shape) == 0:
            L = torch.ones_like(tgt) * L
        # L = L.clip(-5.0, 5.0)
        # loss = torch.where(
        #     L >= 0.0,
        #     torch.log(1.0 + torch.exp(-L)) + (1.0 - tgt) * L,
        #     torch.log(1.0 + torch.exp(L)) - tgt * L,
        # )
        loss = F.binary_cross_entropy_with_logits(L, tgt, reduction='none')
        loss = (loss * M).sum()
        if self.spatial:
            loss = loss + self.sp.penalty()
        loss = loss / tgt.shape[0]
        return L, loss
    
    def loss_mse(
        self,
        Y: Tensor,
        C: Optional[Tensor] = None,
        A: Optional[Tensor] = None,
        M: Optional[Tensor] = None,
    ):
        if M is None:
            M = torch.ones_like(Y)
        Yhat = self.forward(C, A) * M
        # return Yhat, 0.0
        sig = (1e-4 + self.log_sig).exp()
        prec = 1.0 / (1e-4 + sig ** 2)
        err = 0.5 * prec * F.mse_loss(Yhat, Y, reduction="none") + self.log_sig
        prior = sig - self.log_sig
        # err = 0.5 * F.mse_loss(Yhat, Y, reduction="none") * M
        loss = ((err * M).sum() + prior)
        # sigerr = F.smooth_l1_loss(mse.detach().m,sqrt(), sig)
        loss = loss
        if self.spatial:
            loss = loss + self.sp.penalty()
        loss = loss  / Y.shape[0]
        return Yhat, loss

    def loss(self, *args, **kwargs):
        if self.loss_type == "binary":
            l = self.loss_binary(*args, **kwargs)
        elif self.loss_type == "mse":
            l = self.loss_mse(*args, **kwargs)
        return l 


class ResLocalMLP(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        n_hidden: int = 16,
        depth: int = 3,
        groups: int = 1,
        dropout: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        assert n_hidden % groups == 0
        assert depth >= 2
        self.first = nn.Sequential(
            nn.Conv2d(cin, n_hidden, kernel_size=1, bias=True),
            LayerNorm(n_hidden),
        )
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()
        self.middle = nn.ModuleList()
        for _ in range(depth - 2):
            layer = conv_block(n_hidden, n_hidden, 1, bottleneck=False, groups=groups)
            self.middle.append(layer)
        self.final = nn.Conv2d(n_hidden, cout, kernel_size=1)

    def forward(self, inputs: Tensor):
        x = F.GELU(self.first(inputs))
        if len(self.middle) > 1:
            for f in self.middle[:-1]:
                x = F.GELU(x + f(x))
        x = self.dropout(x)
        if len(self.middle) > 0:
            f = self.middle[-1]
            x = F.GELU(x + f(x))
        x = self.final(x)
        return x


class PooledRegression:
    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray = None,
        lam: float = 1.0,
        sample_lam: bool = True,
        seasonal: float = False,
    ):
        self.Y = Y
        self.X = X
        self.seasonal = seasonal
        self.has_covariates = (X is not None)

        self.nt, self.m = Y.shape  # number of reps, number of sites

        #  opts
        self.sample_lam = sample_lam
        self._sampling = True
        self.a = 1.0
        self.b = 1.0

        if self.has_covariates:
            self.d = X.shape[1]
            self.XtXsum = np.einsum("tdn, tfn -> df", self.X, self.X)
            self.lam = np.full((self.d,), lam)
        else:
            self.lam = 0.0
            self.sample_lam = False


        # model params
        self.alpha = 0.0
        self.beta = np.zeros((self.d,)) if self.has_covariates else 0.0
        self.prec = 1.0
        self.ns = 12
        self.ny = self.nt // self.ns
        self.delta = np.zeros((self.ny,))
        self.gamma = np.zeros((self.ns,))
        self.s = np.arange(self.nt) % self.ns
        self.y = np.arange(self.nt) // self.ns
        self.Mu = self.predict()

    def predict(self, X=None):
        if X is None:
            X = self.X
        return (
            np.einsum("tdn,d->tn", X, self.beta) if self.has_covariates else 0.0
            + self.alpha
            + self.delta[self.y, None]
            + self.gamma[self.s, None]
        )

    def sampling(self):
        self._sampling = True

    def optimizing(self):
        self._sampling = False

    def prec_step(self):
        sse = np.square(self.Mu - self.Y)
        an = self.a + 0.5 * self.nt * self.m
        bn = self.b + 0.5 * sse.sum()
        if self._sampling:
            self.prec = np.random.gamma(an, 1.0 / bn)
        else:
            self.prec = (an - 1.0) / bn
        self.prec = np.clip(self.prec, 0.01, 1e6)

    def lam_step(self):
        ss = np.square(self.beta)
        an = self.a + 0.5
        bn = self.b + 0.5 * ss
        if self._sampling:
            self.lam = np.random.gamma(an, 1.0 / bn)
        else:
            self.lam = (an - 1.0) / bn

    def beta_step(self):
        old_beta_contrib = np.einsum("tdn,d->tn", self.X, self.beta)
        resid = self.Y - self.Mu + old_beta_contrib
        XtR = np.einsum("tdn,tn->d", self.X, resid)
        Q = self.XtXsum.copy()
        Q[np.diag_indices_from(Q)] += self.lam
        Sig = np.linalg.inv(Q)
        self.beta = Sig @ XtR
        if self._sampling:
            self.beta += np.random.multivariate_normal(np.zeros(self.d), Sig)
        # new_beta_contrib = np.einsum("tdn,d->tn", self.X, self.beta)
        # self.Mu += new_beta_contrib - old_beta_contrib
        self.Mu = self.predict()

    def delta_step(self):
        old_delta_contrib = self.delta[self.y, None]
        aux = self.Y - self.Mu + old_delta_contrib
        for i in range(1, self.ny):
            self.delta[i] = aux[
                i * self.ns : min(self.nt, (i + 1) * self.ns)
            ].mean()
            if self._sampling:
                self.delta[i] += np.random.randn() / np.sqrt(
                    self.prec * self.ns * self.m
                )
        # self.Mu += self.delta[self.y, None] - old_delta_contrib
        self.Mu += self.predict()

    def gamma_step(self):
        old_gamma_contrib = self.gamma[self.s, None]
        aux = self.Y - self.Mu + old_gamma_contrib
        for i in range(1, self.ns):
            self.gamma[i] = aux[i :: self.ns].mean()
            if self._sampling:
                self.gamma[i] += np.random.randn() / np.sqrt(
                    self.prec * self.ny * self.m
                )
        # self.Mu += self.gamma[self.s, None] - old_gamma_contrib
        self.Mu = self.Mu = self.predict()

    def alpha_step(self):
        # old_alpha = self.alpha
        # resid = self.Y - self.Mu + self.alpha
        # self.alpha = resid.mean()
        self.alpha = self.Y.mean()
        # if self._sampling:
        #     self.alpha += np.random.randn() / np.sqrt(
        #         self.prec * self.nt * self.m
        #     )
        # self.Mu += self.alpha - old_alpha
        self.Mu = self.predict()

    def mcmc_step(self):
        try:
            self.alpha_step()
            if self.has_covariates:
                # self.beta_step()
                pass
            self.delta_step()
            if self.ns > 0 and self.seasonal:
                # self.gamma_step()
                pass
            self.prec_step()
            if self.sample_lam and self.has_covariates:
                # self.lam_step()
                pass
        except:
            print(f"Error in mcmc_step, prec is {self.prec}")
            raise

    def opt_step(self):
        self.optimizing()
        self.mcmc_step()
        self.sampling()
