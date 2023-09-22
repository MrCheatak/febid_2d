"""
Module for running series of experiments using multiprocessing.
"""

import copy
import os
import time
import timeit
import pickle
import traceback

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from processclass import Experiment2D
from program import loop_param, plot
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process


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
    map_2d_mp(pr, (param_name1, param_name2), (vals1, vals2))


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
    map_2d_mp(pr, (param_name1, param_name2), (vals1, vals2))


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
    pr.step = 0.1  # nm
    param_name1 = 'D'
    param_name2 = 'f0'
    vals1 = np.power(np.arange(0, 14100+2, 2), 2)
    # vals1 = [0]
    vals2 = np.arange(1e3, 8e7+3.5e4, 1e4)
    # vals2 = np.power(10, vals2).astype(int)
    # exps_all = []
    start = 7000
    map_2d_mp(pr, (param_name1, param_name2), (vals1, vals2), n_threads=2, init_i=start)


def map_r4():
    """
    Generates raw data for mapping on a tau/p_o grid.
    Iterated variables: sqrt(D), f0

    Resolutions:
        tau_r - [1, 520, 0.3]
        p_o - [0, 10, 0.007]
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
    pr.fwhm = 50  # nm
    pr.order = 1
    pr.step = 0.1  # nm
    param_name1 = 'D'
    param_name2 = 'f0'
    vals1 = np.power(np.arange(0, 7000+5, 5), 2)
    # vals1 = [0]
    vals2 = np.arange(1e3, 2e7+1e4, 1e4)
    # vals2 = np.power(10, vals2).astype(int)
    # exps_all = []
    start = 64
    map_2d(pr, (param_name1, param_name2), (vals1, vals2), n_threads=2, init_i=start)


def map_r5():
    """
    Generates raw data for mapping on a tau/p_o grid.
    Iterated variables: sqrt(D), f0

    Resolutions:
        tau_r - [1, 520, 0.3]
        p_o - [0, 10, 0.007]
    :return:
    """
    # Setting up
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
    pr.fwhm = 50  # nm
    pr.order = 1
    pr.step = 0.1  # nm
    pr.beam_type = 'super_gauss'
    pr.nn = 4
    param_name1 = 'D'
    param_name2 = 'f0'
    vals1 = np.power(np.arange(0, 7000+5, 5), 2)
    # vals1 = [0]
    vals2 = np.arange(1e3, 2e7+1e4, 1e4)
    # vals2 = np.power(10, vals2).astype(int)
    # exps_all = []
    start = 302
    map_2d(pr, (param_name1, param_name2), (vals1, vals2), n_threads=4, init_i=start)


def track_progress(args):
    """
    Track progress for each argument.
    Creates stacked progress bars in console.

    Should be launched as subprocess and terminated explicitly.
    There should be no other output to console.

    :param args: a tuple of current iteration and total N of iterations of a tracked task
    :return:
    """
    pbs = [tqdm(total=args[i][1], initial=args[i][0], position=i, leave=False, ncols=80, desc=f'Process {i}') for i in range(len(args))]
    pbs.append(tqdm(total=args[0][1]*args[1][1], initial=args[0][0]*args[1][1], ncols=160, position=len(pbs), desc='Total progress'))
    pbs[0].desc = 'Main'
    while True:
        for i in range(len(pbs)-1):
            if pbs[i].n < args[i][0]:
                pbs[i].n = args[i][0]
            if pbs[i].n >= pbs[i].total:
                pbs[i].reset()
                pbs[i].n = 0
            pbs[i].refresh()
            time.sleep(1e-2)
        i += 1
        pbs[i].n = pbs[0].n*pbs[1].total + np.asarray([pb.n for pb in pbs[1:i]]).sum()
        pbs[i].refresh()


def map_2d_mp(pr, names, vals, fname='exps', n_threads=4, init_i=0):
    """
    Run scan across two base parameters and dump results to disk.
    Uses multiple CPU cores.

    :param pr: Experiment instance with initial conditions
    :param names: two parameter names
    :param vals: two value collections
    :param fname: base name for files
    :param n_threads: number of CPU cores to use
    :param init_i: skip values of the first parameter
    :return:
    """
    print(f'Running mapping task. Scanning across {names[0]} and {names[1]} parameters. \n'
          f'Value ranges: {names[0]}: {vals[0][0]}-{vals[0][-1]}, {names[1]}: {vals[1][0]}-{vals[1][-1]} \n'
          f'Grid size: {vals[0].size} * {vals[1].size} = {vals[0].size * vals[1].size} virtual experiments.')
    if init_i:
        print(f'Continuing from Experiments Set No.{init_i}: {names[0]}={vals[0][init_i]}\n')
    ###Testing
    backend = 'gpu'
    ###


    start = timeit.default_timer()
    # Per process progress tracking setup
    mgr = Manager()
    progress = mgr.list()
    [progress.append([0, 0]) for i in range(n_threads+1)]
    l = progress[0]
    l[1] = vals[0].size
    l[0] = init_i
    progress[0] = l
    progress_process = Process(target=track_progress, args=[progress])

    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(init_i, vals[0].size):
            val = vals[0][i]
            pr1 = copy.deepcopy(pr)
            pr1.__setattr__(names[0], val)
            # if i % n_threads//2 == 0:
            #     backend = 'cpu'
            # else:
            #     backend = 'gpu'
            fut = executor.submit(dispatch, names[1], vals[1], pr1, backend, progress)
            # fut.add_done_callback(progress_callback)
            fut.__setattr__('id', i)
            futures.append(fut)
        progress_process.start()
        for fut in as_completed(futures):
            try:
                exps = fut.result()
            except Exception:
                tb = traceback.format_exc()
                print(f'Encountered an error while processing Experiments Set {fut.id}: {fut.exception().args}, {fut.exception().with_traceback(tb)}, skipping')
                traceback.print_tb()
            directory = 'sim_data'
            filename = f'{fname}_{fut.id}.obj'
            filepath = os.path.join(directory, filename)
            exps.save_to_file(filepath)
            # with open(filepath, mode='wb') as f:
            #     pickle.dump(exps, f)
            #     print(f'Wrote {fut.id}')
            l = progress[0]
            l[0] += 1
            progress[0] = l
        progress_process.terminate()
    dt = timeit.default_timer() - start
    print(f'Took {dt:.3f} s')


def map_2d(pr, names, vals, fname='exps', n_threads=1, init_i=0):
    """
    Run scan across two base parameters and dump results to disk.
    Uses multiple CPU cores.

    :param pr: Experiment instance with initial conditions
    :param names: two parameter names
    :param vals: two value collections
    :param fname: base name for files
    :param n_threads: number of CPU cores to use
    :param init_i: skip values of the first parameter
    :return:
    """
    print(f'Running mapping task. Scanning across {names[0]} and {names[1]} parameters. \n'
          f'Value ranges: {names[0]}: {vals[0][0]}-{vals[0][-1]}, {names[1]}: {vals[1][0]}-{vals[1][-1]} \n'
          f'Grid size: {vals[0].size} * {vals[1].size} = {vals[0].size * vals[1].size} virtual experiments.')
    if init_i:
        print(f'Continuing from Experiments Set No.{init_i}: {names[0]}={vals[0][init_i]}\n')
    ###Testing
    backend = 'gpu'
    ###

    # Per process progress tracking setup
    mgr = Manager()
    progress = mgr.list()
    [progress.append([0, 0]) for i in range(1+1)]
    l = progress[0]
    l[1] = vals[0].size
    l[0] = init_i
    progress[0] = l
    progress_process = Process(target=track_progress, args=[progress])
    progress_process.start()

    start = timeit.default_timer()

    for i in range(init_i, vals[0].size):
        val = vals[0][i]
        pr1 = copy.deepcopy(pr)
        pr1.__setattr__(names[0], val)
        # if i % n_threads//2 == 0:
        #     backend = 'cpu'
        # else:
        #     backend = 'gpu'
        try:
            exps = dispatch(names[1], vals[1], pr1, backend, progress)
        except Exception as e:
            tb = traceback.format_exc()
            print(f'Encountered an error while processing Experiments Set {i}: {e.args}, {e.with_traceback(tb)}, skipping')
            traceback.print_tb()
            continue
        directory = 'sim_data'
        filename = f'{fname}_{i}.obj'
        filepath = os.path.join(directory, filename)
        exps.save_to_file(filepath)
        l = progress[0]
        l[0] += 1
        progress[0] = l
    progress_process.terminate()
    dt = timeit.default_timer() - start
    print(f'Took {dt:.3f} s')


if __name__ == '__main__':
    dir = 'sim_data'
    map_r4()

