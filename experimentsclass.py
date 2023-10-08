from copy import deepcopy, copy
from multiprocessing import current_process
from pickle import dump
import numpy as np
import matplotlib.pyplot as plt

from processclass import Experiment2D
from analyse import get_peak, deposit_fwhm


class ExperimentSeries2D:
    """
    Class containing a series of experiments resulting from changing a single parameter.
    """
    _experiments = None
    n_experiments = 0
    param = None

    def __init__(self):
        self._experiments = []

    def add_experiment(self, pr):
        """
        Append an experiment to the series.

        :param pr: experiment instance
        """
        self._experiments.append(pr)
        self.n_experiments += 1

    def get_attr(self, name):
        """
        Extract a parameter or a result feature from all the experiments.
        :param name: name of the parameter
        :return:
        """
        try:
            vals_list = [getattr(exp, name) for exp in self._experiments]
        except AttributeError:
            try:
                vals_list = getattr(self, name)
            except AttributeError:
                raise AttributeError(f'Attribute \'{name}\' not found!.')
        vals_arr = np.array(vals_list)
        return vals_arr

    @property
    def r_max(self):
        """
        Get positions of the growth speed maximum in all experiments.
        Peaks are considered symmetric if they not at 0.
        :return:
        """
        r_max = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            r_max[i] = exp.r_max
        return r_max

    @property
    def R_max(self):
        """
        Get peak growth rate in all experiments.
        :return:
        """
        R_max = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            R_max[i] = exp.R_max
        return R_max

    @property
    def R_0(self):
        """
        Get growth rate at the center of the BIR in all experiments.
        :return:
        """
        R_center = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            R_center[i] = exp.R_0
        return R_center

    def get_peak_twinning(self):
        """
        Get a boolean array for if the growth rate has two maximums.
        :return:
        """
        twin_peak = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            maxima = exp.r_max
            twin_peak[i] = (maxima > 1 and maxima > exp.step)
        return twin_peak

    @property
    def fwhm_d(self):
        """
        Get growth rate peak size (FWHM) from all experiments.
        :return:
        """
        fwhm = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            r, R = exp.r, exp.R
            fwhm[i] = deposit_fwhm(r, R)
        return fwhm

    @property
    def R_ind(self):
        """
        Get relative indent of the growth rate profiles with two maximums.
        :return:
        """

        R_ind = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            R_ind[i] = exp.R_ind
        return R_ind

    def plot(self, var='R', label=True):
        if var=='R':
            y_label = 'R/sJV'
        elif var == 'n':
            y_label = 'n'
        else:
            print(f'Variable {var} not found.')
            return
        fig, ax = plt.subplots()
        label_text = None
        for pr in self._experiments:
            if label:
                fwhm = deposit_fwhm(pr.r, pr.R)
                label_text = f'{self.param}={getattr(pr, self.param):.1e}\n'
                label_text += f'p_i={pr.p_i:.3f} ' \
                        f'ρ_o={pr.p_o:.3f} ' \
                        f'τ={pr.tau_r:.3f}\n' \
                        f'φ={fwhm / pr.fwhm:.3f} ' \
                        f'φ1={pr.phi1:.3f} ' \
                        f'φ2={pr.phi2:.3f}\n'
            _ = ax.plot_from_exps(pr.r, getattr(pr, var), label=label_text)
        ax.set_ylabel(y_label)
        ax.set_xlabel('r')
        plt.legend(fontsize=6, loc='upper right')
        plt.show()

    def save_to_file(self, filename):
        """
        Save experiments to a file.
        :param filename: full file name (including path and extension)
        :return:
        """
        with open(filename, mode='wb') as f:
            dump(self, f)


    def __getitem__(self, key):
        return deepcopy(self._experiments[key])


def loop_param(name, vals, pr_init: Experiment2D, backend='cpu', mgr=None, **kwargs):
    """
    Iterate over a specified parameter, solve numerically and collect resulting experiments.
    Multiprocessing-safe.

    :param name: name of the iterated parameters
    :param vals: values to iterate
    :param pr_init: initial conditions
    :param backend: compute on 'cpu' or 'gpu'
    :param mgr: multiprocessing progress tracker
    :return: ExperimentSeries2D
    """
    if mgr is not None:
        cp = current_process()
        cp_id = cp._identity
        if len(cp_id) < 1:
            # This is Main Process
            cp_id = 1
        else:
            # Child process
            cp_id = cp._identity[0] - 2
        l = mgr[cp_id]
        l[0] = 1
        l[1] = vals.size
        mgr[cp_id] = l
    pr = copy(pr_init)
    pr.backend = backend
    r = pr.get_grid()
    f = pr.get_beam(r)
    n_a = pr.analytic(r)
    exps = ExperimentSeries2D()
    for val in vals:
        setattr(pr, name, val)
        if name in ['fwhm', 'f0', 'st_dev']:
            bonds = pr.get_bonds()
            r = np.arange(-bonds, bonds, pr.step)
            f = pr.get_beam(r)
            n_a = pr.analytic(r)
        pr.solve_steady_state(r, f, n_init=n_a, **kwargs)
        exps.add_experiment(pr)
        if mgr is not None:
            l = mgr[cp_id]
            l[0] += 1
            mgr[cp_id] = l
    exps.param = name
    return exps
