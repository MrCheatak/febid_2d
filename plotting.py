import numpy as np
import matplotlib.pyplot as plt


def plot_map(x, y, z, title=None, xlabel=r'$p_{out}$', ylabel=r'$tau_r$', logx=False, logy=False, dpi=300):
    fix, ax = plt.subplots(figsize=(12.8, 9.6), dpi=dpi)
    xp = np.unique(x)
    yp = np.unique(y)
    zz = z.reshape(yp.size, xp.size)
    img = ax.imshow(zz, cmap='magma', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.semilogx()
    if logy:
        ax.semilogy()
    plt.title(title)
    plt.colorbar(img, ax=ax)
    plt.show()


def plot_from_exps(exps, x_name, y_name, **kwargs):
    x = exps.get_attr(x_name)
    y = exps.get_attr(y_name)
    plot_2d(x, y, kwargs)


def plot_2d(x, y, color=None, title=None, xlabel='x', ylabel='y', logx=False, logy=False, dpi=150):
    fig, ax = plt.subplots(dpi=dpi)
    line = ax.scatter(x, y, c=color, cmap='magma')
    if logx:
        ax.semilogx()
    if logy:
        ax.semilogy()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.show()