import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow_movie(sol, frames=50, t=None, interval=80, cmap='viridis', aspect='equal', figsize=None):
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    fig = plt.figure()
    if figsize is not None:
        fig.set_size_inches(*figsize)
    ax = fig.add_subplot(111)

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    cv0 = sol[0]
    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')

    def animate(frame):
        arr, t = frame
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        tx.set_text(f't={t:.3f}')

    time, w, h = sol.shape
    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate,
                        frames=frames, interval=interval)

    return HTML(ani.to_jshtml())


def plotline_movie(sol, frames=50, t=None, X=None, interval=80, ylim=None):
    sol = np.asarray(sol)
    if len(sol.shape) == 2:
        sol = np.expand_dims(sol, axis=0)
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()

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
    # return ani
    return HTML(ani.to_jshtml())
