from math import sqrt, log
import numpy as np


class BeamSettings:
    """
    Class for beam settings.

    Attributes
    ----------
    st_dev : float
        Standard deviation of the Gaussian beam.
    fwhm : float
        FWHM of the Gaussian beam, nm.
    beam_type : str
        Type of the beam. Can be 'gauss' or 'super_gauss'.
    order : int
        Super-Gaussian order.
    """
    def __init__(self):
        self.beam_settings = ['st_dev', 'fwhm', 'beam_type', 'order']
        self._st_dev = 1
        self._fwhm = 2.355 * self._st_dev
        self._beam_type = 'gauss'
        self.order = 1
        self._load_var_definitions()

    @property
    def st_dev(self):
        """
        Standard deviation of the Gaussian beam.
        :return:
        """
        fwhm = self._fwhm
        n = 1
        if self.beam_type == 'super_gauss':
            n = self.order
        self._st_dev = fwhm / (2 * sqrt(2) * (log(2)) ** (1 / 2 / n))
        return self._st_dev

    @st_dev.setter
    def st_dev(self, val):
        self._st_dev = val
        _ = self.fwhm

    @property
    def fwhm(self):
        """
        FWHM of the Gaussian beam, nm.
        :return:
        """
        s = self._st_dev
        n = 1
        if self.beam_type == 'super_gauss':
            n = self.order
        self._fwhm = s * 2 * sqrt(2) * (log(2)) ** (1 / 2 / n)
        return self._fwhm

    @fwhm.setter
    def fwhm(self, val):
        self._fwhm = val
        _ = self.st_dev

    @property
    def beam_type(self):
        """
        Type of the beam. Can be 'gauss' or 'super_gauss'.
        :return:
        """
        return self._beam_type

    @beam_type.setter
    def beam_type(self, val):
        self._beam_type = val

    def generate_beam_profile(self, x):
        """
        Generate beam profile on the given grid.
        :param x: 1d array of x values
        :return: 1d array
        :return:
        """
        if self.beam_type == 'gauss':
            return self._gauss(x)
        elif self.beam_type == 'super_gauss':
            return self._super_gauss(x)
        else:
            raise ValueError('Beam type not recognized')

    def print_beam_settings(self):
        self._print_params(self.beam_settings)

    def _print_params(self, params):
        text = ''
        for param in params:
            val = getattr(self, param)
            val_format = self.__custom_format(val)
            text += param + ': ' + val_format + '\n'
        print(text)

    @staticmethod
    def __custom_format(self, number):
        if type(number) is not type(None) or str:
            if number > 1000:
                return "{:.3e}".format(number)
            elif number >= 0.01:
                return "{:.3g}".format(number)
            elif number < 0.01:
                return "{:.3e}".format(number)
        else:
            return str(number)

    def _gauss(self, x):
        return self.__gauss_expression(x, n=1)

    def _super_gauss(self, x):
        return self.__gauss_expression(x, n=self.order)

    def _load_var_definitions(self):
        """
        Definitions of the variables.
        :return:
        """
        text = self._local_var_defs()
        if hasattr(self, 'param_defs') is False:
            self.param_defs = text
        else:
            self.param_defs += text

    def _local_var_defs(self):
        text=''
        try:
            text += super()._local_var_defs()
        except AttributeError:
            pass
        text += ('Beam settings:\n'
                'st_dev: Standard deviation of the (Super-)Gaussian function describing the beam shape.\n'
                'fwhm: Full Width at Half Maximum of the beam, nm.\n'
                'beam_type: Type of the beam. Can be "gauss" or "super_gauss".\n'
                'order: Super-Gaussian order.\n')
        return text

    def help(self):
        print(self.param_defs)

    def __gauss_expression(self, x, n):
        return np.exp(-0.5 * (((x / self.st_dev) ** 2) ** (n)))

