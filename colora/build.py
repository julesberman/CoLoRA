import flax.linen as nn
import jax
from jax.flatten_util import ravel_pytree

from colora.networks import DNN
from colora.utils import init_net, merge, split


def build_colora(u_hat_config, h_config, x_dim, mu_t_dim, u_dim, lora_filter='alpha', period=None, rank=1, full=False):

    net_u = DNN(**u_hat_config, out_dim=u_dim,
                period=period, rank=rank, full=full)
    params_init, u_apply = init_net(net_u, x_dim)

    # split up the params of u into the offline params (theta) and the online params (phi)
    phi_init, theta_init = split(params_init, lora_filter)
    flat_phi, phi_unravel = ravel_pytree(phi_init)
    n_phi = len(flat_phi)

    net_h = DNN(**h_config, out_dim=n_phi)
    psi_init, h_fn = init_net(net_h, mu_t_dim)

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
