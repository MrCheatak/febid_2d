import numpy as np
from math import sqrt, log
from copy import deepcopy
import matplotlib.pyplot as plt
from multiprocessing import current_process
from tqdm import tqdm

from experimentsclass import ExperimentSeries2D
from analyse import deposit_fwhm

n0 = 3.0  # 1/nm^2
F = 2000.0  # 1/nm^2/s
s = 1.0
V = 0.4  # nm^3
tau = 200e-6  # s
D = 2e5  # nm^2/s
sigma = 0.02  #
f0 = 1.0e7
st_dev = 20.0
order = 1.0

fwhm = 2.355 * st_dev

kd = s * F / n0 + 1 / tau + sigma * f0
kr = s * F / n0 + 1 / tau

nd = s * F / kd
nr = s * F / kr

tau_in = 1 / kd
tau_out = 1 / kr
tau_r = tau_out / tau_in

p_in = sqrt(D * tau_in)
p_out = 2 * p_in / fwhm

phi_1 = sqrt(log(1 + tau_r, 2))
phi_2 = sqrt(log(2 + p_out ** -2))

start = -log(f0, 10) / 4 * st_dev * 3
end = -start
step = 0.3

dt_des = tau
dt_diss = 1 / sigma / f0
dt_diff = step ** 2 / (2 * D)
dt = np.min([dt_des, dt_diss, dt_diff]) * 0.7

n = 0.0
f = 1e5
n_D = 1e-5

# fig, ax = plt.subplots()
#
#
# def plot_line(x, y, name=None, marker=None):
#     global ax
#     line, = ax.plot(x, y, label=name, marker=marker)
#     plt.legend(fontsize=6, loc='upper right')
#     # plt.pause(0.2)
#     return line


# def show(pr, title=None):
#     global ax
#     position = (0.02, 0.76)
#     text = f'n0={pr.n0}\n' \
#            f's={pr.s}\n' \
#            f'F={pr.F}\n' \
#            f'tau={pr.tau:.2e}\n' \
#            f'D={pr.D:.2e}\n' \
#            f'sigma={pr.sigma}\n' \
#            f'f0={pr.f0:.2e}\n' \
#            f'fwhm={pr.fwhm}'
#     plt.text(*position, text, transform=ax.transAxes, fontsize=6, snap=True)
#     plt.title(label=title, loc='center')
#     plt.show()


def plot(exps, x_name, y_name, title=None, color=None, logx=False, logy=False):
    fig, ax = plt.subplots()
    x = exps.get_attr(x_name)
    y = exps.get_attr(y_name)
    line = ax.scatter(x, y, c=color, cmap='magma')
    if logx:
        ax.semilogx()
    if logy:
        ax.semilogy()
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.title(title)
    plt.show()

def plot_freex(exps, x, y_name, x_name=None, title=None, color=None, logx=False, logy=False):
    fig, ax = plt.subplots()
    y = exps.get_attr(y_name)
    x = np.array(x)
    line = ax.scatter(x, y, c=color, cmap='magma')
    if logx:
        ax.semilogx()
    if logy:
        ax.semilogy()
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.title(title)
    plt.show()

def plot_freey(exps, x_name, y, y_name=None, title=None, color=None, logx=False, logy=False):
    fig, ax = plt.subplots()
    x = exps.get_attr(x_name)
    y = np.array(y)
    line = ax.scatter(x, y, c=color, cmap='magma')
    if logx:
        ax.semilogx()
    if logy:
        ax.semilogy()
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.title(title)
    plt.show()


def loop_param(name, vals, pr_init, backend='cpu', mgr=None):
    if mgr:
        cp = current_process()
        cp_id = cp._identity[0]-2
        l = mgr[cp_id]
        l[0] = 1
        l[1] = vals.size
        mgr[cp_id] = l
    pr = deepcopy(pr_init)
    pr.backend = backend
    r = pr.get_grid()
    n_a = pr.analytic(r)
    exps = ExperimentSeries2D()
    for i, val in enumerate(vals):
        # pr = deepcopy(pr)  # copying explicitly to preserve instances in the list
        pr.__setattr__(name, val)
        if name in ['fwhm', 'f0', 'st_dev']:
            bonds = pr.get_bonds()
            r = np.arange(-bonds, bonds, pr.step)
            n_a = pr.analytic(r)
        pr.solve_steady_state(r, n_init=n_a)
        exps.add_experiment(pr)
        if mgr:
            l = mgr[cp_id]
            l[0] += 1
            mgr[cp_id] = l
    exps.param = name
    return exps
