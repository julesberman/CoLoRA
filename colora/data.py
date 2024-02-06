import glob
import os

import h5py
import jax.numpy as jnp
import numpy as np
from einops import rearrange


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
        sols = rearrange(sols, 'M Q T N1 -> M (T N1 Q)')
        X = jnp.expand_dims(x1, axis=-1)
    # two spatial dimensions
    if len(spacing) == 4:
        _, t, x1, x2 = spacing
        sols = rearrange(sols, 'M Q T N1 N2 -> M (T N1 N2 Q)')
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
