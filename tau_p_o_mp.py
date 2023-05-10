import copy
import os
import timeit

import numpy as np
import matplotlib.pyplot as plt
from processclass import Experiment2D
from program import loop_param, plot, plot_r_max
from main import get_r_max, get_peak, get_peak_twinning, get_r_maxes

from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor, as_completed
from multiprocessing import Lock, Queue


pr = Experiment2D()
# Initializing model
pr.n0 = 2.7  # 1/nm^2
pr.F = 730.0  # 1/nm^2/s
pr.s = 1.0
pr.V = 0.05  # nm^3
pr.tau = 100e-6  # s
pr.D = 8e5  # nm^2/s
pr.sigma = 0.02  # nm^2
pr.f0 = 1.0e7
pr.fwhm = 50  # nm
pr.order = 1
pr.step = 0.5  # nm


def progress_callback(future):
    print('.', end='', flush=True)


def write_to_file(p_o, tau_r, r_max):
    fname = 'tau_p_4.txt'
    if not os.path.exists(fname):
        with open(fname, mode='x') as f:
            f.write('tau_r\tp_o\tr_max\n')
            for i in range(p_o.size):
                f.write(f'{p_o[i]}\t{tau_r[i]}\t{r_max[i]}\n')
    else:
        with open(fname, mode='a') as f:
            for i in range(p_o.size):
                f.write(f'{p_o[i]}\t{tau_r[i]}\t{r_max[i]}\n')
    print('Wrote!')


def dispatch(*args):
    exps = loop_param(*args)
    # exps_all.append(exps)

    return exps


if __name__ == '__main__':
    # Setting up
    param_name1 = 'D'
    param_name2 = 'f0'
    vals1 = np.power(np.arange(0, 3000, 20), 2)
    vals2 = np.arange(2e5, 5e7, 2e5)
    # vals2 = np.power(10, vals2).astype(int)
    data = np.zeros((vals1.shape[0] * vals2.shape[0], 3))
    # exps_all = []
    temp = pr.__getattribute__(param_name1)
    start = timeit.default_timer()
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, val in enumerate(vals1):
            pr1 = copy.deepcopy(pr)
            pr1.__setattr__(param_name1, val)
            fut = executor.submit(dispatch, param_name2, vals2, pr1)
            fut.add_done_callback(progress_callback)
            futures.append(fut)
        for f in as_completed(futures):
            exps = f.result()
            r_max = exps.get_peak_position()
            r_max_normed = 2 * r_max / exps.get_attr('fwhm')
            tau_r = exps.get_attr('tau_r')
            p_o = exps.get_attr('p_o')
            # with lock:
            write_to_file(p_o, tau_r, r_max_normed)
    dt = timeit.default_timer() - start
    print(f'Took {dt:.3f} s')
    pr.__setattr__(param_name1, temp)

