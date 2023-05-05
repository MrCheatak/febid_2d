from copy import deepcopy
import numpy as np

from processclass import Experiment2D
from analyse import get_peak, deposit_fwhm


class ExperimentSeries2D:
    _experiments = None
    n_experiments = 0

    def __init__(self):
        self._experiments = []

    def add_experiment(self, pr: Experiment2D):
        self._experiments.append(deepcopy(pr))
        self.n_experiments += 1
        return self._experiments

    def get_attr(self, name):
        vals_list = [exp.__getattribute__(name) for exp in self._experiments]
        vals_arr = np.array(vals_list)
        return vals_arr

    def get_peak_position(self):
        r_max = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            r, R = exp.r, exp.R
            maxima, _ = get_peak(r, R)
            r_max[i] = maxima.max()
        return r_max

    def get_peak_twinning(self):
        twin_peak = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            r, R = exp.r, exp.R
            maxima, _ = get_peak(r, R)
            twin_peak[i] = (maxima.shape[0] > 1 and maxima.max() > exp.step)
        return twin_peak

    def get_deposit_fwhm(self):
        fwhm = np.zeros(self.n_experiments)
        for i, exp in enumerate(self._experiments):
            r, R = exp.r, exp.R
            fwhm[i] = deposit_fwhm(r, R)
