import random

import flax
import haiku as hk
import jax
import jax.numpy as jnp


def init_net(net, input_dim, key=None):
    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 10_000))
    pt = jnp.zeros(input_dim)
    theta_init = net.init(key, pt)
    f = net.apply
    return theta_init, f


def split(theta_phi, filter_list):

    if not isinstance(filter_list, list):
        filter_list = [filter_list]

    def filter_rn(m, leaf_key, p):
        return leaf_key in filter_list

    _, theta_phi = flax.core.pop(theta_phi, 'params')
    phi, theta = hk.data_structures.partition(filter_rn, theta_phi)

    phi = {'params': phi}
    theta = {'params': theta}
    return phi, theta


def merge(phi, theta):
    theta_phi = hk.data_structures.merge(phi['params'], theta['params'])
    theta_phi = {'params': theta_phi}
    return theta_phi
