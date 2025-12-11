import numpy as np
from pickle import dump
from backend.beam_settings import BeamSettings


class ContinuumModel(BeamSettings):
    """
    The class represents single precursor Continuum Model with diffusion.
    It contains all parameters including base precursor, beam and process parameters.
    Process parameters such as depletion or replenishment rate are calculated according to the model.
    The class also provides an appropriate time step for numerical solution.
    """
    def __init__(self):
        super().__init__()
        self.name = ''
        self.base_params = ['s', 'F', 'n0', 'tau', 'sigma', 'f0', 'D', 'V']
        self.s = 1.0
        self.F = 1.0  # 1/nm^2/s
        self.n0 = 1.0  # 1/nm^2
        self.tau = 1.0  # s
        self.sigma = 1.0  # nm^2
        self.f0 = 1.0  # 1/(nm^2*s)
        self.D = 0.0  # nm^2/s
        self.V = 1.0  # nm^3

        self._N = 100
        self._step = 1.0  # nm

        self._dt = 1.0
        self._dt_diff = np.nan
        self._dt_des = np.nan
        self._dt_diss = np.nan

        self.process_attrs = ['kd', 'kr', 'nd', 'nr', 'tau_in', 'tau_out', 'tau_r', 'p_in', 'p_out', 'p_i', 'p_o',
                              'phi1', 'phi2']
        self._kd = np.nan
        self._kr = np.nan
        self._nd = np.nan
        self._nr = np.nan
        self._tau_in = np.nan
        self._tau_out = np.nan
        self._tau_r = np.nan
        self._p_in = np.nan
        self._p_out = np.nan
        self._p_i = np.nan
        self._p_o = np.nan
        self._phi1 = np.nan
        self._phi2 = np.nan

    @property
    def N(self):
        """
        Reaction region discretization.
        Calculated as FWHM_B/step.
        :return:
        """
        return self._N

    @N.setter
    def N(self, val):
        self._N = val
        self._step = self.fwhm / self._N

    @property
    def step(self):
        """
        Grid step size, nm.
        :return:
        """
        return self._step

    @step.setter
    def step(self, val):
        self._step = val
        self._N = int(self.fwhm / self._step)

    @property
    def dt(self):
        """
        Maximal time step for the solution of the reaction-diffusion equation, s.
        :return:
        """
        self._dt = np.min([self.dt_des, self.dt_diss, self.dt_diff]) * 0.5
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
        Relative depletion or just Delpetion. Defined as ratio between effective residence time in the center and outside the beam.
        :return:
        """
        self._tau_r = self.tau_out / self.tau_in
        return self._tau_r

    @property
    def p_in(self):
        """
        Precursor molecule diffusion path in the center of the beam, nm.
        :return:
        """
        self._p_in = np.sqrt(self.D * self.tau_in)
        return self._p_in

    @property
    def p_out(self):
        """
        Precursor molecule diffusion path outside the beam, nm.
        :return:
        """
        self._p_out = np.sqrt(self.D * self.tau_out)
        return self._p_out

    @property
    def p_i(self):
        """
        Normalized precursor molecule diff. path in the center of the beam.
        :return:
        """
        self._p_i = 2 * self.p_in / self.fwhm
        return self._p_i

    @property
    def p_o(self):
        """
        Diffusive replenishment. Normalized precursor molecule diff. path outside the beam.
        :return:
        """
        self._p_o = 2 * self.p_out / self.fwhm
        return self._p_o

    @property
    def phi1(self):
        """
        Deposit size relative to beam size without surface diffusion.
        First scaling law. Applies only to gaussian beams.
        :return:
        """
        self._phi1 = np.power(np.log2(1 + self.tau_r), 1 / 2 / self.order)
        return self._phi1

    @property
    def phi2(self):
        """
        Deposit size relative to beam size with surface diffusion.
        Second scaling law. Applies only to gaussian beams.
        :return:
        """
        if self.p_o != 0:
            self._phi2 = np.power(np.log2(2 + (self.tau_r - 1) / (1 + self.p_o ** 2)), 1 / 2 / self.order)
        else:
            self._phi2 = np.nan
        return self._phi2

    def print_initial_parameters(self):
        self._print_params(self.base_params)

    def print_process_attributes(self):
        self._print_params(self.process_attrs)

    def save_to_file(self, filename):
        """
        Save experiment to a file.
        :param filename: full file name (including path and extension)
        :return:
        """
        with open(filename, mode='wb') as f:
            dump(self, f)

    def _local_var_defs(self):
        """
        Definitions of the variables.
        :return:
        """
        text=''
        try:
            text += super()._local_var_defs()
        except AttributeError:
            pass
        text += ('\nPrecursor parameters:\n'
                's: Precursor sticking coefficient, 1/nm^2.\n'
                'F: Precursor surface flux, 1/nm^2/s.\n'
                'n0: Maximum precursor coverage, 1/nm^2.\n'
                'tau: Residence time of the precursor, s.\n'
                'sigma: Dissociation cross-section of the precursor, nm^2.\n'
                'f0: Surface electron flux density, 1/nm^2/s.\n'
                'D: Diffusion coefficient of the precursor, nm^2/s.\n'
                'V: Volume of a deposited fraction of the precursor molecule, nm^3.\n'
                '\nProcess parameters:\n'
                'kd: Depletion rate, Hz.\n'
                'kr: Replenishment rate, Hz.\n'
                'nd: Depleted precursor coverage, 1/nm^2.\n'
                'nr: Replenished precursor coverage, 1/nm^2.\n'
                'tau_in: Effective residence time in the center of the beam, s.\n'
                'tau_out: Effective residence time outside the beam, s.\n'
                'tau_r: Relative depletion or just Delpetion. Defined as ratio between effective residence time in the center and outside the beam.\n'
                'p_in: Precursor molecule diffusion path in the center of the beam, nm.\n'
                'p_out: Precursor molecule diffusion path outside the beam, nm.\n'
                'p_i: Normalized precursor molecule diff. path in the center of the beam.\n'
                'p_o: Diffusive replenishment. Normalized precursor molecule diff. path outside the beam.\n'
                'phi1: Deposit size relative to beam size without surface diffusion. First scaling law. Applies only to gaussian beams.\n'
                'phi2: Deposit size relative to beam size with surface diffusion. Second scaling law. Applies only to gaussian beams.\n'
                'step: Grid resolution, nm.\n')
        return text


if __name__ == '__main__':
    var = ContinuumModel()
    var.F = 0
