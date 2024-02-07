from typing import List, Optional

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.random import KeyArray

from colora.networks import DNN
from colora.utils import init_net, merge, split


def build_colora(
    u_hat_config: dict,
    h_config: dict,
    x_dim: int,
    mu_t_dim: int,
    u_dim: int,
    lora_filter: List[str] = ['alpha'],
    period: Optional[jnp.ndarray] = None,
    rank: int = 1,
    full: bool = False,
    key: Optional[KeyArray] = None
):
    """Function to set up the full CoLoRA architecture both the hypernetwork given by h and the reduce model given by u_hat

    Args:
        u_hat_config (dict): configuration for u_hat such as layers, width, etc....
        h_config (dict): configuration for h such as layers, width, etc....
        x_dim (int): dim of spatital domain
        mu_t_dim (int): dim of mu + 1 for the time dimensions
        u_dim (int): dim of output domain, usually equal to number of variables in PDE
        lora_filter (List[str], optional): list of keys by which to seperate out the phi parameters from the theta parameters in the u_hat network. Defaults to ['alpha'].
        period (Optional[jnp.ndarray], optional): _description_. Defaults to None.
        rank (int, optional): rank of AB. Defaults to 1.
        full (bool, optional): whether number of alphas per colora layer equals rank of AB (True) or is set to one (False). Defaults to False.
        key (_type_, optional): jax random key, if None will create on own . Defaults to None.

    Returns:
        u_hat: callable function giving the reduced model u_hat where u_hat(theta, phi, x) -> solution
        h: callable function giving the hyper network where h(psi, mu_t) -> phi
        theta_init: parameter initializations for theta
        psi_init: parameter initializations for psi
    """
    net_u = DNN(**u_hat_config, out_dim=u_dim,
                period=period, rank=rank, full=full)
    params_init, u_apply = init_net(net_u, x_dim, key=key)

    # split up the params of u into the offline params (theta) and the online params (phi)
    phi_init, theta_init = split(params_init, lora_filter)
    flat_phi, phi_unravel = ravel_pytree(phi_init)
    n_phi = len(flat_phi)

    net_h = DNN(**h_config, out_dim=n_phi)
    psi_init, h_fn = init_net(net_h, mu_t_dim, key=key)

    # now we define a wrapper over the u neural network
    # this will allow us to take theta and phi sepearatly
    # and then it will automatically merge them and pass the resulting
    # combined paramters to u_apply
    # it also unravels phi so that we can pass in phi as a vector that is output from h

    def build_u_hat(u_apply, phi_unravel):
        def u_hat(theta, phi, *args):
            phi = phi_unravel(phi)
            theta_phi = merge(theta, phi)
            return u_apply(theta_phi, *args)
        return u_hat

    u_hat_fn = build_u_hat(u_apply, phi_unravel)

    return u_hat_fn, h_fn, theta_init, psi_init
