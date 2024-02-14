
from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers


class Periodic(nn.Module):

    width: int
    period: Optional[jnp.ndarray]
    param_dtype = jnp.float32
    with_bias: bool = True
    w_init: Callable = initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        dim, f = x.shape[-1], self.width
        w_init = self.w_init
        period = jnp.asarray(self.period)

        a = self.param('a', w_init, (f, dim), self.param_dtype)
        phi = self.param('c', w_init, (f, dim), self.param_dtype)

        omeg = jnp.pi*2/period
        o = a*jnp.cos(omeg*x+phi)
        if self.with_bias:
            b = self.param('b', w_init, (f, dim), self.param_dtype)
            o += b

        o = jnp.mean(o, axis=1)

        return o
    
class Random_Freq(nn.Module):

    features: int
    param_dtype = jnp.float32
    variance: int = None

    @nn.compact
    def __call__(self, x):
        f, dim = self.features, x.shape[-1]
        a_init = initializers.normal(stddev=self.variance)

        R = self.param('R', a_init, (f, dim), self.param_dtype)
        fs = f//2

        s = jnp.sin(R[:fs] @ x)
        c = jnp.cos(R[fs:] @ x)
        return jnp.concatenate([s, c])


class CoLoRA(nn.Module):

    width: int
    rank: int
    full: bool
    w_init: Callable = initializers.lecun_normal()
    b_init: Callable = initializers.zeros_init()
    with_bias: bool = True
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, X):
        D, K, r = X.shape[-1], self.width, self.rank

        w_init = self.w_init
        b_init = self.b_init
        z_init = initializers.zeros_init()

        W = self.param('W', w_init, (D, K), self.param_dtype)
        A = self.param('A', w_init, (D, r), self.param_dtype)
        B = self.param('B', z_init, (r, K), self.param_dtype)

        if self.full:
            n_alpha = self.rank
        else:
            n_alpha = 1

        alpha = self.param('alpha', z_init, (n_alpha,), self.param_dtype)

        AB = (A*alpha)@B
        AB = AB  # / r
        W = (W + AB)

        out = X@W

        if self.with_bias:
            b = self.param("b", b_init, (K,))
            b = jnp.broadcast_to(b, out.shape)
            out += b

        return out
