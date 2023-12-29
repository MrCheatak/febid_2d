import numpy as np
import numexpr_mod as ne

from processclass import Experiment2D


class Experiment2D_Dimensionless(Experiment2D):
    """
    Class representing a virtual experiment with beam settings and dimensionless parameters.
    It allows calculation of a 2D precursor coverage and growth rate profiles based on the set conditions.
    """
    def __init__(self, backend='cpu'):
        super().__init__()
        self.backend = backend
        self._numexpr_name = 'r_e'
        self._step = np.nan
        self.tau_r = 1.0
        self.p_o = 1.0
        self.step = 1.0
        self.fwhm = 1.0
        self.f = 0.0
        k = 0.0
        n = 0.0
        n_D = 0.0
        local_dict = self._local_dict
        ne.cache_expression("(1 - k * n + p_o**2*n_D/step**2)*dt + n", self._numexpr_name,
                            local_dict=local_dict)

        self.s = np.nan
        self.F = np.nan
        self.n0 = np.nan
        self.tau = np.nan
        self.sigma = np.nan
        self.D = np.nan
        self.V = np.nan
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
        print('GPU backend not implemented, defaulting to CPU.')
        self.backend = 'cpu'
        return self.__numeric(*args, **kwargs)

    def __copy__(self):
        pr = Experiment2D_Dimensionless()
        pr.f0 = self.f0
        pr.fwhm = self.fwhm
        pr.beam_type = self.beam_type
        pr.order = self.order
        pr.tau_r = self.tau_r
        pr.p_o = self.p_o
        pr.step = self.step
        pr.backend = self.backend
        return pr


if __name__ == '__main__':
    from analyse import deposit_fwhm as df
    pr_d = Experiment2D_Dimensionless()

    pr_d.tau_r = 20
    pr_d.p_o = 0.6
    pr_d.f0 = 9e5
    pr_d.beam_type = 'super_gauss'
    pr_d.order = 4
    pr_d.fwhm = 1400
    pr_d.step = pr_d.fwhm // 200
    pr_d.solve_steady_state(progress=True)
    pr_d.plot('R')
    print([pr_d.fwhm_d, pr_d.fwhm])
    print([pr_d.fwhm_d/pr_d.fwhm, pr_d.phi2])
    print([2*pr_d.r_max/pr_d.fwhm, pr_d.R_ind1])
    a = 0


