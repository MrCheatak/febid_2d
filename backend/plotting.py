import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_map(x, y, z, title=None, xlim=None, ylim=None, xlabel=r'$p_{out}$', ylabel=r'$tau_r$', logx=False, logy=False,
             colormap='magma', vmin=None, vmax=None, contour=False, levels=None, levels_labels=True, manual_locations=None,
             contour_font=None, contour_format=None, colors='k', figsize=(12.8, 9.6), dpi=300):
    fix, ax = plt.subplots(figsize=figsize, dpi=dpi)
    xp = np.unique(x)
    yp = np.unique(y)
    # zz = z.reshape(yp.size, xp.size)
    # if not xlim:
    #     xlim = [x.min(), x.max()]
    # if not ylim:
    #     ylim = [y.min(), y.max()]
    img = ax.imshow(z, cmap=colormap, vmin=vmin, vmax=vmax, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
    if contour:
        def fmt(val):
            return f'{val:.{contour_format}f}'
        cont = ax.contour(xp, yp, z, levels, colors=colors, linewidths=0.3)
        if levels_labels:
            ax.clabel(cont, cont.levels, inline=True, fmt=fmt, fontsize=contour_font, manual=manual_locations)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if logx:
        ax.semilogx()
    if logy:
        ax.semilogy()
    plt.title(title)
    plt.colorbar(img, ax=ax)
    plt.show()
    return fix, ax


def plot_from_exps(exps, x_name, y_name, **kwargs):
    """
    Plot 2D data by the name from a collection of experiments.
    :param exps: collection of experiments
    :param x_name: variable name for x-axis
    :param y_name: variable name for y-axis
    :param kwargs: for additional arguments refer to plot_2d
    :return:
    """
    x = exps.get_attr(x_name)
    y = exps.get_attr(y_name)
    plot_2d(x, y, **kwargs)


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


# Helper funcs
def plot_img(x, y, z, title='Surface map'):
    """
    Create a color-mapped surface projection. Each coordinate must be a matrix.

    Uses matplotlib.
    :param x:
    :param y:
    :param z:
    :param title:
    :return:
    """
    # The extent parameter is set to the range of coordinates in the X and Y matrices.
    # The aspect='auto' parameter ensures that the aspect ratio of the plot is adjusted appropriately.
    plt.imshow(z, cmap='magma', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Show the plot
    plt.show()


def plot_surf_3d(x, y, z, title='Surface map'):
    """
    Create a 3D color-mapped surface. Each coordinate must be a matrix.

    Uses Plotly.
    :param x:
    :param y:
    :param z:
    :param title:
    :return:
    """
    fig1 = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig1.update_layout(title=title, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig1.show()


def plot_scat_2d(x, y, z, title='Scatter plot'):
    """
    Create a color-mapped scatter plot. Each coordinate must be 1D.

    Uses matplotlib.
    :param x:
    :param y:
    :param z:
    :param title:
    :return:
    """
    fix, ax = plt.subplots(figsize=(12.8, 9.6), dpi=70)
    line = ax.scatter(x, y, c=z, s=0.5, cmap='magma')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title(title)
    plt.colorbar(line, ax=ax)
    plt.show()


def plot_map_plotly(x, y, z):
    """
    Plot a map of z on the grid of x and y.
    :param x: 1d array of size m, ticks at X-axis
    :param y: 1d array of size n, ticks on Y-axis
    :param z: 2d array of size m*n, data to be plotted
    :return:
    """
    fig = go.Figure()
    layout = {
        'width': 800,
        'height': 800,
        'autosize': False,
    }
    heatmap = go.Figure(go.Heatmap(z=z, x=x, y=y), layout=layout)
    # heatmap.show()
    return heatmap
