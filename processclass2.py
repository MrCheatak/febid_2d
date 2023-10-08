import math
import numpy as np
import numexpr_mod as ne
import pyopencl as cl
from tqdm import tqdm

from processclass import Experiment2D


class Experiment2D_Dimensionless(Experiment2D):

    def __init__(self, backend='cpu'):
        super().__init__()
        self.backend = backend
        self._numexpr_name = 'r_e'
        self._step = np.nan
        self.tau_r = 1
        self.p_o = 1
        self.step = 1
        self.fwhm = 1
        self.f = 0.0
        k = 0.0
        n = 0.0
        n_D = 0.0
        local_dict = self._local_dict
        ne.cache_expression("(1 - k * n + p_o**2*n_D/step**2)*dt + n", self._numexpr_name,
                            local_dict=local_dict)

    @property
    def _local_dict(self):
        k = (self.tau_r - 1) / self.f0 * self.f + 1
        return dict(k=k, p_o=self.p_o, step=self.step*2/self.fwhm, dt=self.dt)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, val):
        """
        Set grid resolution in nm.
        :param val:
        :return:
        """
        self._step = val

    @property
    def R(self):
        self._R = self.n * (self._local_dict['k']-1)
        return self._R

    @property
    def dt(self):
        return min(self.dt_des_diss, self.dt_diff) * 0.7

    @dt.setter
    def dt(self, val):
        self._dt = val

    @property
    def dt_des_diss(self):
        return 1 / self.tau_r

    @property
    def dt_diff(self):
        if self.p_o > 0:
            return (self.step / self.fwhm) ** 2 / (2 * self.p_o ** 2)
        else:
            return 1

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

    def analytic(self, r, f, tau_r=None, p_o=None, out=None):
        if tau_r is None:
            tau_r = self.tau_r
        if p_o is None:
            p_o = self.p_o
        k = (self.tau_r - 1) / self.f0 * f + 1
        n = 1 / k
        if out is not None:
            out[:] = n[:]
        else:
            self.n = n
        return n

    def __numeric_gpu(self, *args, **kwargs):
        return self.__numeric(*args, **kwargs)
