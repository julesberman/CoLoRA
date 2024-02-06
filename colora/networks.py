from collections.abc import Iterable
from typing import Callable, List

import flax.linen as nn
import jax
import jax.numpy as jnp

from colora.layers import CoLoRA, Periodic


class DNN(nn.Module):
    width: int
    layers: List[str]
    out_dim: int
    activation: Callable = jax.nn.swish
    period: float = None
    rank: int = 1
    full: int = False

    @nn.compact
    def __call__(self, x):
        depth = len(self.layers)
        width = self.width

        A = self.activation
        for i, layer in enumerate(self.layers):
            is_last = i == depth - 1

            if isinstance(self.activation, Iterable):
                A = self.activation[i]

            if is_last:
                width = self.out_dim
            L = get_layer(layer=layer, width=width,
                          period=self.period, rank=self.rank, full=self.full)
            x = L(x)
            if not is_last:
                x = A(x)

        return x


def get_layer(layer, width, period=None, rank=1, full=False):
    if layer == 'D':
        L = nn.Dense(width)
    elif layer == 'P':
        L = Periodic(width, period=period)
    elif layer == 'C':
        L = CoLoRA(width, rank, full)
    else:
        raise Exception(f"unknown layer: {layer}")
    return L
