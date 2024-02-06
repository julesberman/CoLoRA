import random
from functools import partial

import jax
import numpy as np
import optax
from jax import jit
from tqdm.auto import tqdm

str_to_opt = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'adamww': partial(optax.adamw, weight_decay=1e-3),
    'amsgrad':  optax.amsgrad,
    'adabelief': optax.adabelief,
}


def adam_opt(theta_init, loss_fn, args=(), init_state=None, steps=1000, learning_rate=1e-3, scheduler=True, verbose=False, loss_tol=None, optimizer='adam'):
    # adds warm up cosine decay
    if scheduler:
        learning_rate = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=steps,
            alpha=0.0
        )

    opti_f = str_to_opt[optimizer]
    optimizer = opti_f(learning_rate=learning_rate)

    state = optimizer.init(theta_init)
    if init_state is not None:
        state = init_state

    @jit
    def step(params, state, args):
        loss_value, grads = jax.value_and_grad(
            loss_fn)(params, *args)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return loss_value, params, state

    params, opt_params = theta_init, theta_init
    pbar = tqdm(range(steps), disable=not verbose)
    opt_loss = float('inf')
    loss_history = []
    n_rec = max(steps // 1000, 1)

    for i in pbar:

        cur_loss, params_new, state_new = step(params, state, args)

        pbar.set_postfix({'loss': f'{cur_loss:.3E}'})

        if cur_loss < opt_loss:
            opt_loss = cur_loss
            opt_params = params_new

        if i % n_rec:
            loss_history.append(cur_loss)

        params = params_new
        state = state_new

        if loss_tol is not None and cur_loss < loss_tol:
            break

    loss_history = np.asarray(loss_history)
    return opt_params, loss_history
