from processclass import Experiment2D
from analyse import get_peak
import program
import numpy as np

pr = Experiment2D()

pr.n0 = 2.7  # 1/nm^2
pr.F = 730.0  # 1/nm^2/s
pr.s = 1.0
pr.V = 0.05  # nm^3
pr.tau = 200e-6  # s
pr.D = 8e5  # nm^2/s
pr.sigma = 0.02  # nm^2
pr.f0 = 1.0e7
pr.fwhm = 50  # nm
pr.order = 1
pr.step = 0.5  # nm



def get_r_max(r, R):
    N = len(R)
    r_max = np.zeros(N)
    for i in range(N):
        r_max[i], _ = get_peak(r[i], R[i])
    return r_max


def get_r_maxes(exps):
    r_max = np.zeros(len(exps))
    twin_peak = np.zeros_like(r_max)
    for i, exp in enumerate(exps):
        maxima, _ = get_peak(exps[i].r, exps[i].R)
        r_max[i] = maxima.max()
        twin_peak[i] = (maxima.shape[0]>1 and maxima.max()>exp.step)
    # twin_peak = get_peak_twinning(r_max)
    return r_max, twin_peak


def get_peak_twinning(x):
    twin_peak = np.array([r.shape[0] == 2 for r in x])
    return twin_peak


def r_max_vs_D(n0=None, s=None, F=None, tau=None, sigma=None, f0=None, fwhm=None):
    for arg_name, arg_val in locals().items():
        if arg_val is not None:
            pr.__setattr__(arg_name, arg_val)
    vals = [0, *np.arange(3.75e5, 3.8e5, 0.01e5), 3.8e5, 3.85e5, 3.9e5, 3.95e5, 4e5, 4.45e5, 5e5, 6e5, 7e5, 1e6, 1.5e6, 2e6]
    val_name = 'D'
    temp = pr.__getattribute__(val_name)
    n = program.loop_param(val_name, vals, pr)
    pr.__setattr__(val_name, temp)
    r_max = get_r_max(pr.r, pr.R)
    program.fig, program.ax = program.plt.subplots()
    program.plot_line(vals, r_max)
    program.show(pr, 'Peak position vs diffusion coeff.')
    a=0
    return pr.r, pr.R, n, r_max


def r_max_vs_f0(n0=None, s=None, F=None, tau=None, D=None, sigma=None, fwhm=None):
    for arg_name, arg_val in locals().items():
        if arg_val is not None:
            pr.__setattr__(arg_name, arg_val)
    val_name = 'f0'
    vals = np.arange(5, 8.5, 0.2)
    vals = np.power(10, vals).astype(int)
    temp = pr.__getattribute__(val_name)
    n = program.loop_param(val_name, vals, pr)
    pr.__setattr__(val_name, temp)
    r_max = get_r_max(pr.r,pr.R)
    program.fig, program.ax = program.plt.subplots()
    program.plot_line(np.log10(vals), r_max)
    program.show(pr, 'Peak position vs electron flux')
    a=0
    return pr.r, pr.R, n, r_max


def r_max_vs_fwhm(n0=None, s=None, F=None, tau=None, D=None, sigma=None, f0=None):
    for arg_name, arg_val in locals().items():
        if arg_val is not None:
            pr.__setattr__(arg_name, arg_val)
    val_name = 'fwhm'
    vals = np.arange(5, 100, 7)
    temp = pr.__getattribute__(val_name)
    n = program.loop_param(val_name, vals, pr)
    pr.__setattr__(val_name, temp)
    r_max = get_r_max(pr.r,pr.R)
    program.fig, program.ax = program.plt.subplots()
    program.plot_line(vals, r_max)
    program.show(pr, 'Peak position vs beam FWHM')
    a=0
    return pr.r, pr.R, n, r_max


def r_max_vs_tau(n0=None, s=None, F=None, D=None, sigma=None, f0=None, fwhm=None):
    for arg_name, arg_val in locals().items():
        if arg_val is not None:
            pr.__setattr__(arg_name, arg_val)
    val_name = 'tau'
    vals = np.arange(50e-6, 500e-6, 20e-6).round(6)
    temp = pr.__getattribute__(val_name)
    n = program.loop_param(val_name, vals, pr)
    pr.__setattr__(val_name, temp)
    r_max = get_r_max(pr.r,pr.R)
    program.fig, program.ax = program.plt.subplots()
    program.plot_line(vals, r_max)
    program.show(pr, 'Peak position vs residence time')
    a=0
    return pr.r, pr.R, n, r_max


def r_max_vs_sigma(n0=None, s=None, F=None, tau=None, D=None, f0=None, fwhm=None):
    for arg_name, arg_val in locals().items():
        if arg_val is not None:
            pr.__setattr__(arg_name, arg_val)
    val_name = 'sigma'
    vals = np.arange(0.002, 0.03, 0.002).round(6)
    temp = pr.__getattribute__(val_name)
    n = program.loop_param(val_name, vals, pr)
    pr.__setattr__(val_name, temp)
    r_max = get_r_max(pr.r, pr.R)
    program.fig, program.ax = program.plt.subplots()
    program.plot_line(vals, r_max)
    program.show(pr, 'Peak position vs diss. cross-section')
    a=0
    return pr.r, pr.R, n, r_max



if __name__ == '__main__':
    pr.tau_r
    r_max_vs_D()
    # r_max_vs_f0()
    # r_max_vs_tau()



