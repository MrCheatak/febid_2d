import math
import numpy as np
import numexpr_mod as ne
import pyopencl as cl
from tqdm import tqdm

from processclass import Experiment2D


class PseudoExperiment2D(Experiment2D):
    r_unnormed = None
    f_unnormed = None

    def __init__(self, backend='cpu'):
        super().__init__()
        self.backend = backend
        self._numexpr_name = 'r_e'
        n = 0.0
        n_D = 0.0
        f = 0.0
        r = 0.0
        local_dict = dict(tau_r=self.tau_r, p_o=self.p_o, dt=self.dt, fwhm=self.fwhm)
        ne.cache_expression("(1 - (tau_r - 1) * 2**(-(r*p_o*fwhm/2)**2) * n +n_D)*dt + n", self._numexpr_name,
                            local_dict=local_dict)

        self.tau_r = 1
        self.p_o = 1

    @property
    def _local_dict(self):
        return dict(tau_r=self.tau_r, p_o=self.p_o, dt=self.dt, fwhm=self.fwhm)

    def get_grid(self, bonds=None):
        if not bonds:
            bonds = self.get_bonds()
        self.r_unnormed = np.arange(-bonds, bonds, self.step)
        self.r = self.r_unnormed / self.p_o
        return self.r

    def get_gauss(self, r):
        self.f = np.exp(-r ** 2 / (2 * self.st_dev ** 2))
        self.f_unnormed = self.f * self.f0
        return self.f

    @property
    def R(self):
        self._R = self.n * self.f * (self.tau_r - 1)
        return self._R

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = val

    @property
    def tau_r(self):
        return self._tau_r

    @tau_r.setter
    def tau_r(self, val):
        self._tau_r = val

    @property
    def p_o(self):
        return self._p_o

    @p_o.setter
    def p_o(self, val):
        self._p_o = val
