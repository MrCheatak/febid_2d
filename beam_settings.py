from math import sqrt, log


class BeamSettings:

    def __init__(self):
        self.beam_settings = ['st_dev', 'fwhm', 'beam_type', 'order']
        self._st_dev = 1
        self._fwhm = 2.355 * self._st_dev
        self.beam_type = 'gauss'
        self.order = 1

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

    def print_beam_settings(self):
        self._print_params(self.beam_settings)

    def _print_params(self, params):
        text = ''
        for param in params:
            val = getattr(self, param)
            val_format = self.__custom_format(val)
            text += param + ': ' + val_format + '\n'
        print(text)

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
