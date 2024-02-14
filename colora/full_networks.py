

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from flax.linen import initializers
from jax import numpy as jnp
from jax import random, vmap

from colora.layers import Periodic, Random_Freq


class Fourier_Layer(nn.Module):

    frequency: int
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        m, dx, dy = x.shape
        F = self.frequency
        w_init = initializers.glorot_normal()

        phi_a = self.param('phi_a', w_init, (m, F, F), self.param_dtype)
        phi_b = self.param('phi_b', w_init, (m, F, F), self.param_dtype)

        ff = jnp.fft.fft2(x).real
        f_top = ff[:, :F, :F]
        f_bot = ff[:, -F:, -F:]

        # point wise multiply
        f_top *= phi_a
        f_bot *= phi_b

        ff = ff.at[:, :F, :F].set(f_top)
        ff = ff.at[:, -F:, -F:].set(f_bot)

        y = jnp.fft.ifft2(ff).real

        return y


class Fourier_Layer_Lora(nn.Module):

    frequency: int
    rank: int
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        m, dx, dy = x.shape
        F, r = self.frequency, self.rank
        w_init = initializers.glorot_normal()

        phi_a = self.param('phi_a', w_init, (m, F, F), self.param_dtype)
        alpha = self.param('alpha', initializers.ones,
                           (1), self.param_dtype)

        ff = jnp.fft.rfft2(x).real

        ff = jnp.fft.fftshift(ff)
        f_top = ff[:, :F, :F]

        # point wise multiply
        f_top *= phi_a*alpha[0]

        ff = ff.at[:, :F, :F].set(f_top)

        ff = jnp.fft.ifftshift(ff)
        y = jnp.fft.irfft2(ff).real

        return y


class CNN(nn.Module):
    features: int
    depth: int
    quantities: int
    frequency: int
    variance: int = 1
    rank: int = 1
    period: int = None

    @nn.compact
    def __call__(self, X):
        N = len(X)
        X = vmap(Periodic(width=self.features, period=self.period))(X)
        X = vmap(Random_Freq(features=self.features, variance=self.variance))(X)
        X = X.reshape(100, 100, self.features)
        # X = rearrange(X, 'nx ny m -> m nx ny')
        X = jnp.expand_dims(X, axis=0)
        for d in range(self.depth-1):
            # C = Fourier_Layer_Lora(
            #     frequency=self.frequency, rank=self.rank)
            C = SpectralConv2d(out_channels=self.features,
                               modes1=self.frequency, modes2=self.frequency)
            X = C(X)
            X = jax.nn.swish(X)

        X = rearrange(X, 'b nx ny f -> (b nx ny) f')
        X = nn.Dense(self.quantities)(X)
        return X


def normal(stddev=1e-2, dtype=jnp.float32) -> Callable:
    def init(key, shape, dtype=dtype):
        keys = random.split(key)
        return random.normal(keys[0], shape) * stddev
    return init


class SpectralConv2d(nn.Module):
    out_channels: int = 32
    modes1: int = 12
    modes2: int = 12
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # x.shape: [batch, height, width, in_channels]

        # Initialize parameters
        in_channels = x.shape[-1]
        scale = 1/(in_channels * self.out_channels)

        in_channels = x.shape[-1]
        height = x.shape[1]
        width = x.shape[2]

        # Checking that the modes are not more than the input size
        assert self.modes1 <= height//2 + 1
        assert self.modes2 <= width//2 + 1
        assert height % 2 == 0  # Only tested for even-sized inputs
        assert width % 2 == 0  # Only tested for even-sized inputs

        # The model assumes real inputs and therefore uses a real
        # fft. For a 2D signal, the conjugate symmetry of the
        # transform is exploited to reduce the number of operations.
        # Given an input signal of dimesions (N, C, H, W), the
        # output signal will have dimensions (N, C, H, W//2+1).
        # Therefore the kernel weigths will have different dimensions
        # for the two axis.
        kernel_1_r = self.param(
            'kernel_1_r',
            normal(scale, jnp.float32),
            (in_channels, self.out_channels, self.modes1, self.modes2),
            jnp.float32
        )
        kernel_1_i = self.param(
            'kernel_1_i',
            normal(scale, jnp.float32),
            (in_channels, self.out_channels, self.modes1, self.modes2),
            jnp.float32
        )
        kernel_2_r = self.param(
            'kernel_2_r',
            normal(scale, jnp.float32),
            (in_channels, self.out_channels, self.modes1, self.modes2),
            jnp.float32
        )
        kernel_2_i = self.param(
            'kernel_2_i',
            normal(scale, jnp.float32),
            (in_channels, self.out_channels, self.modes1, self.modes2),
            jnp.float32
        )

        alpha = self.param('alpha', initializers.ones,
                           (4), self.param_dtype)

        kernel_1_r *= alpha[0]
        kernel_1_i *= alpha[1]
        kernel_2_r *= alpha[2]
        kernel_2_i *= alpha[3]

        # Perform fft of the input
        x_ft = jnp.fft.rfftn(x, axes=(1, 2))

        # Multiply the center of the spectrum by the kernel
        out_ft = jnp.zeros_like(x_ft)
        s1 = jnp.einsum(
            'bijc,coij->bijo',
            x_ft[:, :self.modes1, :self.modes2, :],
            kernel_1_r + 1j*kernel_1_i)
        s2 = jnp.einsum(
            'bijc,coij->bijo',
            x_ft[:, -self.modes1:, :self.modes2, :],
            kernel_2_r + 1j*kernel_2_i)
        out_ft = out_ft.at[:, :self.modes1, :self.modes2, :].set(s1)
        out_ft = out_ft.at[:, -self.modes1:, :self.modes2, :].set(s2)

        # Go back to the spatial domain
        y = jnp.fft.irfftn(out_ft, axes=(1, 2))

        return y
