class BeamSettings:

    def __init__(self):
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
        self._st_dev = self._fwhm / 2.355
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
        self._fwhm = 2.355 * self._st_dev
        return self._fwhm

    @fwhm.setter
    def fwhm(self, val):
        self._fwhm = val
        _ = self.st_dev
