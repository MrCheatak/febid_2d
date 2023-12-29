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

    def __init__(self, E0, ie, yld, fwhm, order=1, beam_type='gauss'):
        super().__init__()
        self.electron_flux_parameters = ['E', 'ie', 'yld']
        self.E0 = E0  # beam energy, keV
        self.ie = ie  # beam current, A
        self.yld = yld  # secondary electron yield (SEs per PE)
        self.fwhm = fwhm
        self.beam_type = beam_type
        self.order = order

    @property
    def pe_flux(self):
        """
        Total electrons emitted per second.
        :return:
        """
        a = np.round(self.ie / self.e)
        return a

    @property
    def f0_pe(self):
        """
        Pre-exponential factor for primary electron beam.
        """
        if type(self.ie*1.0) is not float and type(self.fwhm*1.0) is not float:
            f0_pe = np.outer(self.pe_flux, 1 / self._gauss_integral)
        else:
            f0_pe = self.pe_flux / self._gauss_integral
        return np.round(f0_pe)

    @property
    def f0_se(self):
        """
        Pre-exponential factor for secondary electron emission profile.
        """
        return np.round(self.f0_pe * self.yld)

    def print_electron_flux_parameters(self, ):
        self._print_params(self.electron_flux_parameters)

    @property
    def _gauss_integral(self):
        """
        Integral of a (super) gaussian function.
        """
        if self.beam_type == "gauss":
            return self.st_dev * sqrt(2 * pi)
        if self.beam_type == 'super_gauss':
            return self.st_dev * sqrt(2) / self.order * gamma(1 / 2 / self.order)


if __name__ == '__main__':
    var = EFluxEstimator(5, 120e-12, 0.67, 100, 2, 'super_gauss')
    var.print_electron_flux_parameters()
