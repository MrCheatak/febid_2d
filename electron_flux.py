from math import sqrt, log10, log, pi
import numpy as np


class EFluxEstimator:
    """
    Class allows estimation of the Secondary electron surface flux based on primary beam current, energy and size
    and material SE yield. Assuming gaussian profile of both PE and SE flux.
    """
    e = 1.602176e-19  # elemental charge, C
    ie = 1e-12  # beam current, A
    E0 = 5  # beam energy, keV
    yld = 0.2  # secondary electron yield (SEs per PE)
    _st_dev = 2.5  # beam gaussian standard deviation
    _fwhm = 2 * log(2) * _st_dev  # beam width, nm

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
        :return:
        """
        return np.int64(self.pe_flux / sqrt(2 * pi) / self.st_dev)

    @property
    def f0_se(self):
        """
        Pre-exponential factor for secondary electron emission profile.
        :return:
        """
        return np.int64(self.f0_pe * self.yld)


