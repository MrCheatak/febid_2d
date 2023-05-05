import numpy as np
from math import sqrt, log


class ContinuumModel:
    """
    Class implementing continuum model for the FEBID process.


    It encapsulates reaction-diffusion equation (RDE) parameters, beam parameters and parameters required for
    numerical solution of the RDE.

    The class calculates all the intermediate parameters describing deposition process features.
    """
    s = 1.0
    F = 1.0  # 1/nm^2/s
    n0 = 1.0  # 1/nm^2
    tau = 1.0  # s
    sigma = 1.0  # nm^2
    f0 = 1.0  # 1/(nm^2*s)
    D = 1.0  # nm^2/s
    V = 1.0  # nm^3

    step = 1.0  # nm

    _st_dev = 1.0
    _fwhm = 1.0

    _dt = 1.0
    _dt_diff = np.nan
    _dt_des = np.nan
    _dt_diss = np.nan

    _kd = np.nan
    _kr = np.nan
    _nd = np.nan
    _nr = np.nan
    _tau_in = np.nan
    _tau_out = np.nan
    _tau_r = np.nan
    _p_in = np.nan
    _p_out = np.nan
    _p_i = np.nan
    _p_o = np.nan
    _phi1 = np.nan
    _phi2 = np.nan


    @property
    def fwhm(self):
        """
        FWHM of the Gaussian beam, nm.
        :return:
        """
        self._fwhm = 2.355 * self._st_dev
        return self._fwhm

    @fwhm.setter
    def fwhm(self, val):
        self._fwhm = val
        _ = self.st_dev

    @property
    def st_dev(self):
        """
        Standard deviation of the Gaussian beam.
        :return:
        """
        self._st_dev = self._fwhm / 2.355
        return self._st_dev

    @st_dev.setter
    def st_dev(self, val):
        self._st_dev = val
        _ = self.fwhm

    @property
    def dt(self):
        """
        Time step of the reaction-diffusion equation, s.
        :return:
        """
        self._dt = np.min([self.dt_des, self.dt_diss, self.dt_diff]) * 0.6
        return self._dt

    @dt.setter
    def dt(self, val):
        dt = self.dt
        if val > dt:
            print(f'Not allowed to increase time step. \nTime step larger than {dt} s will crash the solution.')
        else:
            self._dt = val

    @property
    def dt_diff(self):
        """
        Maximal time step for the diffusion process, s.
        :return:
        """
        if self.D > 0:
            self._dt_diff = self.step ** 2 / (2 * self.D)
        else:
            self._dt_diff = 1
        return self._dt_diff

    @property
    def dt_diss(self):
        """
        Maximal time step for the dissociation process, s.
        :return:
        """
        self._dt_diss = 1 / self.sigma / self.f0
        return self._dt_diss

    @property
    def dt_des(self):
        """
        Maximal time step for the desorption process, s.
        :return:
        """
        self._dt_des = self.tau
        return self._dt_des


    @property
    def kd(self):
        """
        Depletion rate (under beam irradiation), Hz.
        :return:
        """
        self._kd = self.s * self.F / self.n0 + 1 / self.tau + self.sigma * self.f0
        return self._kd

    @property
    def kr(self):
        """
        Replenishment rate (without beam irradiation), Hz.
        :return:
        """
        self._kr = self.s * self.F / self.n0 + 1 / self.tau
        return self._kr

    @property
    def nd(self):
        """
        Depleted precursor coverage (under beam irradiation).
        :return:
        """
        self._nd = self.s * self.F / self.kd
        return self._nd

    @property
    def nr(self):
        """
        Replenished precursor coverage (without beam irradiation).
        :return:
        """
        self._nr = self.s * self.F / self.kr
        return self._nr

    @property
    def tau_in(self):
        """
        Effective residence time in the center of the beam, s.
        :return:
        """
        self._tau_in = 1 / self.kd
        return self._tau_in

    @property
    def tau_out(self):
        """
        Effective residence time outside the beam, s.
        :return:
        """
        self._tau_out = 1 / self.kr
        return self._tau_out

    @property
    def tau_r(self):
        """
        Relative depletion. Defined as ratio between effective residence time in the center and outside the beam.
        :return:
        """
        self._tau_r = self.tau_out / self.tau_in
        return self._tau_r

    @property
    def p_in(self):
        """
        Precursor diffusion path in the center of the beam, nm.
        :return:
        """
        self._p_in = sqrt(self.D * self.tau_in)
        return self._p_in

    @property
    def p_out(self):
        """
        Precursor molecule diffusion path outside the beam, nm.
        :return:
        """
        self._p_out = sqrt(self.D * self.tau_out)
        return self._p_out

    @property
    def p_i(self):
        """
        Diffusive replenishment.
        :return:
        """
        self._p_i = 2 * self.p_in / self.fwhm
        return self._p_i

    @property
    def p_o(self):
        """
        A parameter describing the diffusion length of precursor molecule without being irradiated.
        :return:
        """
        self._p_o = 2 * self.p_out / self.fwhm
        return self._p_o

    @property
    def phi1(self):
        """
        Deposit size relative to beam size under Reaction Rate Limited regime.
        First scaling law.
        :return:
        """
        self._phi1 = sqrt(log(1 + self.tau_r, 2))
        return self._phi1

    @property
    def phi2(self):
        """
        Deposit size relative to beam size under Diffusion Enhanced or Mass Transport Limited regime.
        Second scaling law.
        :return:
        """
        try:
            self._phi2 = sqrt(log(2 + self.p_i ** (-2), 2))
        except ZeroDivisionError:
            self._phi2 = np.nan
        return self._phi2

    @property
    def get_R(self, n, f):
        R = n * self.sigma * f * self.V

    def print_process_attributes(self):
        text = f'' \
               f'kd: {self.kd:.0f}\n' \
               f'kr: {self.kr:.1f}\n' \
               f'nd: {self.nd:.3e}\n' \
               f'nr: {self.nr:.4f}\n' \
               f'tau_in: {self.tau_in:.3e}\n' \
               f'tau_out: {self.tau_out:.3e}\n' \
               f'tau_r: {self.tau_r:.2f}\n' \
               f'p_in: {self.p_in:3f}\n' \
               f'p_out: {self.p_out:3f}\n' \
               f'p_i: {self.p_i:3f}\n' \
               f'p_o: {self.p_o:.3f}\n' \
               f'phi1: {self.phi1:.3f}\n' \
               f'phi2: {self.phi2:.3f}\n'
        print(text)
