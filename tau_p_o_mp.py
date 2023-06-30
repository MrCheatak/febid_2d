import copy
import os
import timeit
import pickle

import numpy as np
import matplotlib.pyplot as plt
from processclass import Experiment2D
from program import loop_param, plot
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor, as_completed


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


def write_to_file(fname, names, *args):
    n = len(args)
    if not os.path.exists(fname):
        with open(fname, mode='x') as f:
            header = '\t'.join(names) + '\n'
            f.write(header)
            text = ''
            for i in range(args[1].size):
                for j in range(n):
                    text += f'{args[j][i]}\t'
                text += '\n'
                f.write(text)
                text = ''
    else:
        with open(fname, mode='a') as f:
            text = ''
            for i in range(args[1].size):
                for j in range(n):
                    text += f'{args[j][i]}\t'
                text += '\n'
                f.write(text)
                text = ''
    print('Wrote!')


def dispatch(*args):
    exps = loop_param(*args)
    return exps


def map_r1():
    """
    Generates raw data for mapping on a tau/p_o grid.
    Iterated variables: sqrt(D), f0

    Resolutions:
        tau_r - [1.389, 97.978, 389]
        p_o - [0, 1.17621, 0.00789]
    :return:
    """
    # Setting up
    fname = 'tau_p_3.txt'
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

    param_name1 = 'D'
    param_name2 = 'f0'
    vals1 = np.power(np.arange(0, 3000, 20), 2)
    vals2 = np.arange(2e5, 5e7, 2e5)
    map_2d(pr, (param_name1, param_name2), (vals1, vals2))


def map_r2():
    """
    Generates raw data for mapping on a tau/p_o grid.
    Iterated variables: sqrt(D), f0

    Resolutions:
        tau_r - [1, 10000, 0.5]
        p_o - [0, 100, 0.00755]
    :return:
    """
    # Setting up
    fname = 'tau_p_6.1.txt'
    pr = Experiment2D()
    # Initializing model
    pr.n0 = 2.7  # 1/nm^2
    pr.F = 730.0  # 1/nm^2/s
    pr.s = 1.0
    pr.V = 0.05  # nm^3
    pr.tau = 2000e-6  # s
    pr.D = 1 # nm^2/s
    pr.sigma = 0.02  # nm^2
    pr.f0 = 1e3
    pr.fwhm = 20  # nm
    pr.order = 1
    pr.step = 0.5  # nm
    param_name1 = 'D'
    param_name2 = 'f0'
    vals1 = np.power(np.arange(2.1*97, 14100+2.1, 2.1), 2)
    vals2 = np.arange(1e3, 1.6e8+7.5e3, 7.5e3)
    map_2d(pr, (param_name1, param_name2), (vals1, vals2))


def map_r3():
    """
    Generates raw data for mapping on a tau/p_o grid.
    Iterated variables: sqrt(D), f0

    Resolutions:
        tau_r - [1, 2000, 0.3]
        p_o - [0, 50, 0.008]
    :return:
    """
    # Setting up
    fname = 'tau_p_8.txt'
    pr = Experiment2D()
    # Initializing model
    pr.n0 = 2.7  # 1/nm^2
    pr.F = 730.0  # 1/nm^2/s
    pr.s = 1.0
    pr.V = 0.05  # nm^3
    pr.tau = 2000e-6  # s
    pr.D = 1 # nm^2/s
    pr.sigma = 0.02  # nm^2
    pr.f0 = 1e3
    pr.fwhm = 20  # nm
    pr.order = 1
    pr.step = 0.5  # nm
    param_name1 = 'D'
    param_name2 = 'f0'
    vals1 = np.power(np.arange(0, 14100+2, 2), 2)
    # vals1 = [0]
    vals2 = np.arange(1e3, 8e7+3.5e4, 1e4)
    # vals2 = np.power(10, vals2).astype(int)
    # exps_all = []
    map_2d(pr, (param_name1, param_name2), (vals1, vals2))


def map_2d(pr, names, vals, fname='exps', n_threads=10):
    temp = pr.__getattribute__(names[0])
    start = timeit.default_timer()
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(0, vals[0].size):
            val = vals[0][i]
            pr1 = copy.deepcopy(pr)
            pr1.__setattr__(names[0], val)
            fut = executor.submit(dispatch, names[1], vals[1], pr1)
            fut.add_done_callback(progress_callback)
            futures.append(fut)
        i = 0
        for f in as_completed(futures):
            exps = f.result()
            directory = 'sim_data'
            fname = f'{fname}_{i}.obj'
            file = os.path.join(directory, fname)
            with open(file, mode='wb') as f:
                pickle.dump(exps, f)
                print(f'Wrote {i}')
            i += 1
    dt = timeit.default_timer() - start
    print(f'Took {dt:.3f} s')
    pr.__setattr__(names[0], temp)

if __name__ == '__main__':
    dir = 'sim_data'
    map_r3()

