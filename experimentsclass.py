from copy import deepcopy
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

    def add_experiment(self, pr: Experiment2D):
        """
        Append an experiment to the series.

        :param pr: experiment instance
        :return:
        """
        self._experiments.append(deepcopy(pr))
        self.n_experiments += 1
        return self._experiments

    def get_attr(self, name):
        """
        Extract a parameter or a result feature from all the experiments.
        :param name: name of the parameter
        :return:
        """
        try:
            vals_list = [exp.__getattribute__(name) for exp in self._experiments]
        except AttributeError:
            try:
                vals_list = self.__getattribute__(name)
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
            r, R = exp.r, exp.R
            maxima, _ = get_peak(r, R)
            r_max[i] = maxima.max()
        return r_max

    @property
    def R_max(self):
        """
        Get peak growth rate in all experiments.
        :return:
        """
        R_max = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            r, R = exp.r, exp.R
            _,maxima = get_peak(r, R)
            R_max[i] = maxima.max()
        return R_max

    @property
    def R_0(self):
        """
        Get growth rate at the center of the BIR in all experiments.
        :return:
        """
        R_center = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            r, R = exp.r, exp.R
            ind = (r==0).nonzero()[0]
            if ind.size == 0:
                ind = np.fabs(r).argmin()
            R_center[i] = R[ind]
        return R_center

    def get_peak_twinning(self):
        """
        Get a boolean array for if the growth rate has two maximums.
        :return:
        """
        twin_peak = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            r, R = exp.r, exp.R
            maxima, _ = get_peak(r, R)
            twin_peak[i] = (maxima.shape[0] > 1 and maxima.max() > exp.step)
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
        R_center = self.R_0
        R_max = self.R_max
        R_ind = (R_max - R_center) / R_max
        return R_ind

    def plot(self, label=True):
        fig, ax = plt.subplots()
        label_text = None
        for pr in self._experiments:
            if label:
                fwhm = deposit_fwhm(pr.r, pr.R)
                label_text = f'{self.param}={pr.__getattribute__(self.param):.1e}\n'
                label_text += f'p_i={pr.p_i:.3f} ' \
                        f'ρ_o={pr.p_o:.3f} ' \
                        f'τ={pr.tau_r:.3f}\n' \
                        f'φ={fwhm / pr.fwhm:.3f} ' \
                        f'φ1={pr.phi1:.3f} ' \
                        f'φ2={pr.phi2:.3f}\n'
            _ = ax.plot(pr.r, pr.R, label=label_text)
        ax.set_ylabel('R/sJV')
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
        return self._experiments[key]
