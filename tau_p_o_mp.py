"""
Module for running series of experiments using multiprocessing.
"""

from copy import copy
import time
import timeit
import traceback

import numpy as np
from tqdm import tqdm
from backend.processclass import Experiment2D
from backend.experimentsclass import loop_param
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process
import pyopencl as cl



def progress_callback(future):
    print('.', end='', flush=True)


def dispatch(*args, **kwargs):
    exps = loop_param(*args, **kwargs)
    return exps


def track_progress(args):
    """
    Track progress for each argument.
    Creates stacked progress bars in console.

    Should be launched as subprocess and terminated explicitly.
    There should be no other output to console.

    :param args: a tuple of current iteration and total N of iterations for each tracked task
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


def map_2d_mp(pr_init, names, vals, fname='exps', n_threads=4, init_i=0, backend='cpu'):
    """
    Run scan across two base parameters and dump results to disk.
    Uses multiple CPU or GPU cores.

    :param pr_init: Experiment instance with initial conditions
    :param names: two parameter names
    :param vals: two value collections
    :param fname: base name for files
    :param n_threads: number of CPU cores to use
    :param init_i: skip values of the first parameter
    :param backend: use 'cpu' or 'gpu'

    :return:
    """
    print(f'Running mapping task. Scanning across {names[0]} and {names[1]} parameters. \n'
          f'Value ranges: {names[0]}: {vals[0][0]}-{vals[0][-1]}, {names[1]}: {vals[1][0]}-{vals[1][-1]} \n'
          f'Grid size: {vals[0].size} * {vals[1].size} = {vals[0].size * vals[1].size} virtual experiments.')
    if init_i:
        print(f'Continuing from Experiments Set No.{init_i}: {names[0]}={vals[0][init_i]}\n')

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
    pr = copy(pr_init)
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(init_i, vals[0].size):
            val = vals[0][i]
            pr.__setattr__(names[0], val)
            # if i % n_threads//2 == 0:
            #     backend = 'cpu'
            # else:
            #     backend = 'gpu'
            fut = executor.submit(dispatch, names[1], vals[1], pr, backend, progress)
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
            filepath = f'{fname}_{fut.id}.obj'
            exps.save_to_file(filepath)
            l = progress[0]
            l[0] += 1
            progress[0] = l
        progress_process.terminate()
    dt = timeit.default_timer() - start
    print(f'Took {dt:.3f} s')


def map_2d(pr_init, names, vals, fname='exps', n_threads=1, init_i=0, backend='cpu'):
    """
    Run scan across two base parameters and dump results to disk.

    :param pr_init: Experiment instance with initial conditions
    :param names: two parameter names
    :param vals: two value collections
    :param fname: base name for files
    :param n_threads: number of CPU cores to use
    :param init_i: skip values of the first parameter
    :param backend: use 'cpu' or 'gpu'

    :return:
    """
    print(f'Running mapping task. Scanning across {names[0]} and {names[1]} parameters. \n'
          f'Value ranges: {names[0]}: {vals[0][0]}-{vals[0][-1]}, {names[1]}: {vals[1][0]}-{vals[1][-1]} \n'
          f'Grid size: {vals[0].size} * {vals[1].size} = {vals[0].size * vals[1].size} virtual experiments.')
    if init_i:
        print(f'Continuing from Experiments Set No.{init_i}: {names[0]}={vals[0][init_i]}\n')

    # Per process progress tracking setup
    progress = None
    mgr = Manager()
    progress = mgr.list()
    [progress.append([0, 0]) for i in range(1+1)]
    l = progress[0]
    l[1] = vals[0].size
    l[0] = init_i
    progress[0] = l
    progress_process = Process(target=track_progress, args=[progress])
    progress_process.start()
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()

    ctx = cl.Context(devices)
    queue = cl.CommandQueue(ctx)
    kwargs = {'ctx': ctx, 'queue': queue}
    pr = copy(pr_init)
    start = timeit.default_timer()

    for i in range(init_i, vals[0].size):
        val = vals[0][i]
        pr.__setattr__(names[0], val)
        # if i % n_threads//2 == 0:
        #     backend = 'cpu'
        # else:
        #     backend = 'gpu'
        try:
            exps = dispatch(names[1], vals[1], pr, backend, progress, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            print(f'Encountered an error while processing Experiments Set {i}: {e.args}, {e.with_traceback(tb)}, skipping')
            traceback.print_tb()
            continue
        filepath = f'{fname}_{i}.obj'
        exps.save_to_file(filepath)
        l = progress[0]
        l[0] += 1
        progress[0] = l
    progress_process.terminate()
    dt = timeit.default_timer() - start
    print(f'Took {dt:.3f} s')


if __name__ == '__main__':
    pass
    fname = r'sim_data\test'
    pr = Experiment2D()
    # Initializing model
    pr.n0 = 2.7  # 1/nm^2
    pr.F = 730.0  # 1/nm^2/s
    pr.s = 1.0
    pr.V = 0.05  # nm^3
    pr.tau = 2000e-6  # s
    pr.D = 0 # nm^2/s
    pr.sigma = 0.02  # nm^2
    pr.f0 = 1e3
    pr.fwhm = 50  # nm
    pr.step = 0.1  # nm
    pr.beam_type = 'super_gauss'
    pr.order = 4
    param_name1 = 'D'
    param_name2 = 'f0'
    vals1 = np.array([1e5, 1e6, 1e7])
    # vals1 = [0]
    vals2 = np.arange(1e3, 8e7+3.5e4, 1e4)
    pr.solve_steady_state(progress=True)

    map_2d(pr, (param_name1, param_name2), (vals1, vals2), fname=fname, backend='gpu')
