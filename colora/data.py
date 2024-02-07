import glob
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax.lib.xla_bridge import get_backend


def read_hd5f_numpy(filepath):
    with h5py.File(filepath, "r") as f:
        dataset = f['u'][()]
        spacing = []
        for d in f['u'].dims:
            spacing.append(f[d.label][()])
        mu = None
        if 'mu' in f.attrs.keys():
            mu = np.asarray(f.attrs['mu'], dtype=np.float32)
        return dataset, spacing, mu


def load_all_hdf5(path):
    mus = []
    sols = []
    files = glob.glob(os.path.join(path, "*.hdf5"))
    for filepath in files:
        sol, space, mu = read_hd5f_numpy(filepath)
        mus.append(mu.ravel())
        sols.append(sol)

    mus = np.asarray(mus)
    sols = np.asarray(sols)

    # first sort by mu
    if mus.shape[1] == 1:
        sort_idx = np.squeeze(np.argsort(mus, axis=0))
        mus, sols = mus[sort_idx], sols[sort_idx]

    return mus, sols, space


def split_data_by_mu(mus, sols, train_mus, test_mus):

    test_indices = np.where(np.isin(mus, test_mus))[0]
    train_indices = np.where(np.isin(mus, train_mus))[0]
    train_sols, test_sols = sols[train_indices], sols[test_indices]

    return train_sols, test_sols


def prepare_coordinate_data(spacing, mus, sols):

    # one spatial dimensions
    if len(spacing) == 3:
        _, t, x1 = spacing
        sols = rearrange(sols, 'M Q T N1 -> (M T) (N1) Q')
        X = jnp.expand_dims(x1, axis=-1)
    # two spatial dimensions
    if len(spacing) == 4:
        _, t, x1, x2 = spacing
        sols = rearrange(sols, 'M Q T N1 N2 -> (M T) (N1 N2) Q')
        m_grids = jnp.meshgrid(x1, x2,  indexing='ij')
        X = jnp.asarray([m.flatten() for m in m_grids]).T

    # ´expand the dimensionality of mu if it is scalar
    if len(mus.shape) == 1:
        mus = jnp.expand_dims(mus, axis=-1)
    t1 = jnp.expand_dims(t, axis=-1)
    mu_t = jnp.concatenate(
        [jnp.repeat(mus, len(t1), axis=0), jnp.tile(t1, (len(mus), 1))],
        axis=1
    )

    return sols, mu_t, X


class Dataset:
    def __init__(self, mu_t, X_grid, y_sol, n_batches, key):

        device = get_backend().platform
        self.x = jax.device_put(mu_t, jax.devices(device)[0])
        self.y = jax.device_put(y_sol, jax.devices(device)[0])
        self.grid = jax.device_put(X_grid, jax.devices(device)[0])

        self.key = key
        self.idx = jnp.arange(0, len(mu_t), 1, dtype=jnp.int32)
        self.n_batches = n_batches

        self.shuffle()
        self.batch()

        self.b_i = 0
        self.epoch = 0
        self.X_grid = X_grid

    def __iter__(self):
        return self

    def batch(self):
        self.bx = jnp.array_split(self.x, self.n_batches)
        self.by = jnp.array_split(self.y, self.n_batches)

    def shuffle(self):
        self.key, skey = jax.random.split(self.key)
        s_idx = jax.random.permutation(skey, self.idx, independent=True)
        self.x = self.x[s_idx]
        self.y = self.y[s_idx]

    def __next__(self):

        if self.b_i >= self.n_batches:
            if self.epoch % 20 == 0:
                self.shuffle()
                self.batch()
            self.b_i = 0
            self.epoch += 1

        x, y = self.bx[self.b_i], self.by[self.b_i]
        X_grid = self.X_grid
        self.b_i += 1

        return x, y, X_grid
