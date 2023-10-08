from math import sqrt, pi
import numpy as np
from scipy.special import gamma

from beam_settings import BeamSettings


class EFluxEstimator(BeamSettings):
    """
    Class allows estimation of the Secondary electron surface flux based on primary beam current, energy and size
    and material SE yield. Assuming gaussian profile of both PE and SE flux.
    """
    e = 1.602176e-19  # elemental charge, C
    ie = 1e-12  # beam current, A
    E0 = 5  # beam energy, keV
    yld = 0.2  # secondary electron yield (SEs per PE)

    def __init__(self, E0, yld, fwhm, order=1, beam_type='gauss'):
        super().__init__()
        self.E0 = E0
        self.yld = yld
        self.fwhm = fwhm
        self.beam_type = beam_type
        self.order = order

    @property
    def pe_flux(self):
        """
        Total electrons emitted per second.
        :return:
        """
        a = np.int64(self.ie / self.e)
        return a

    @property
    def f0_pe(self):
        """
        Pre-exponential factor for primary electron beam.
        """
        return np.int64(self.pe_flux / self.__gauss_integral)

    @property
    def f0_se(self):
        """
        Pre-exponential factor for secondary electron emission profile.
        """
        return np.int64(self.f0_pe * self.yld)

    @property
    def __gauss_integral(self):
        """
        Integral of a (super) gaussian function.
        """
        if self.beam_type == "gauss":
            return self.st_dev * sqrt(2 * pi)
        if self.beam_type == 'super_gauss':
            return self.st_dev / sqrt(2) / self.order * gamma(1 / 2 / self.order)
