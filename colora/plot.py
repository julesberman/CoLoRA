from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow_movie(sol, frames=50, t=None, interval=80, tight=False, title='', cmap='viridis', aspect='equal', live_cbar=False, save_to=None, show=True):

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    cv0 = sol[0]
    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')
    vmax = np.max(sol)
    vmin = np.min(sol)
    ax.set_xticks([])
    ax.set_yticks([])
    if tight:
        plt.tight_layout()

    def animate(frame):
        arr, t = frame
        im.set_data(arr)
        if live_cbar:
            vmax = np.max(arr)
            vmin = np.min(arr)
            im.set_clim(vmin, vmax)
        tx.set_text(f'{title} t={t:.2f}')

    time, w, h = sol.shape
    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate,
                        frames=frames, interval=interval,)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=30)

    if show:
        return HTML(ani.to_jshtml())


def plotline_movie(sol, frames=50, t=None, X=None, interval=80, ylim=None):
    sol = np.asarray(sol)
    if len(sol.shape) == 2:
        sol = np.expand_dims(sol, axis=0)

    n_lines, time, space = sol.shape
    sol = rearrange(sol, 'l t s -> t s l')
    fig, ax = plt.subplots()
    ax.set_ylim([sol.min(), sol.max()])
    if ylim is not None:
        ax.set_ylim(ylim)
    if X is None:
        X = np.arange(sol.shape[1])
    line = ax.plot(X, sol[0])

    def animate(frame):
        sol, t = frame
        ax.set_title(f't={t:.3f}')
        for i, l in enumerate(line):
            l.set_ydata(sol[:, i])
        return line

    def init():
        line.set_ydata(np.ma.array(X, mask=True))
        return line,

    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    sol_frames = sol[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate, frames=frames,
                        interval=interval, blit=True)
    plt.close()
    # return ani
    return HTML(ani.to_jshtml())


def trajectory_movie(y, frames=50, title='', ylabel='', xlabel='Time', legend=[], x=None, interval=80, ylim=None, save_to=None):
    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))

    fig, ax = plt.subplots()
    total = len(x)
    inc = max(total//frames, 1)
    x = x[::inc]
    y = y[::inc]
    if ylim is None:
        ylim = np.array([y.min(), y.max()])
    xlim = [x.min(), x.max()]

    def animate(i):
        ax.cla()
        ax.plot(x[:i], y[:i], marker='o', markevery=[-1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(legend, loc='lower right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} t={x[i]:.2f}')

    ani = FuncAnimation(fig, animate, frames=len(x), interval=interval)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=30)

    return HTML(ani.to_jshtml())
