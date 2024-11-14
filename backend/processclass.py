import math
import numpy as np
import numexpr_mod as ne
import pyopencl as cl
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, make_interp_spline

from backend.pycl_test import cl_boilerplate, reaction_diffusion_jit
from backend.dataclass import ContinuumModel
from backend.analyse import get_peak, deposit_fwhm
from backend.electron_flux import EFluxEstimator
from backend.diffusion import laplace_1d


class Experiment1D(ContinuumModel):
    """
    Class representing a virtual experiment with precursor properties and beam settings.
    It allows calculation of a 2D precursor coverage and growth rate profiles based on the set conditions.
    """

    def __init__(self, backend='cpu'):
        super().__init__()
        self.r = None
        self.f = None
        self.n = None
        self._R = None
        self._interp = None
        self.backend = backend
        self.interpolation_method = 'cubic'
        self._numexpr_name = 'r_e'
        n = 0.0
        n_D = 0.0
        f = 0.0
        local_dict = dict(s=self.s, F=self.F, n0=self.n0, tau=self.tau,
                          sigma=self.sigma, D=self.D, dt=self.dt, step=self.step)
        ne.cache_expression("(s*F*(1-n/n0) - n/tau - sigma*f*n + n_D*D/step**2)*dt + n", self._numexpr_name,
                            local_dict=local_dict)

    @property
    def _local_dict(self):
        return dict(s=self.s, F=self.F, n0=self.n0, tau=self.tau,
                    sigma=self.sigma, D=self.D, dt=self.dt, step=self.step)

    def get_beam(self, r=None):
        """
        Generate electron beam profile based on a Gaussian.
        :param r: radially symmetric grid
        :return:
        """
        if r is None:
            r = self.get_grid()
        self.f = self.f0 * self.generate_beam_profile(r)
        return self.f

    def get_grid(self, bonds=None):
        """
        Generate a radially symmetric grid to perform the calculation.
        :param bonds: custom bonds, a single positive value
        :return:
        """
        if not bonds:
            bonds = self.get_bonds()
        self.r = np.arange(-bonds, bonds, self.step)
        return self.r

    def solve_steady_state(self, r=None, f=None, eps=1e-8, n_init=None, **kwargs):
        """
        Derive a steady state precursor coverage.

        r, f and n_init must have the same length.
        If these parameters are not provided, they are generated automatically.

        If n_init is not provided, an analytical solution is used.

        eps should be changed together with step attribute, otherwise the solution may fall through.

        :param r: radially symmetric grid
        :param f: electron flux
        :param eps: solution accuracy
        :param n_init: initial precursor coverage
        :return: steady state precursor coverage profile
        """
        if r is None:
            r = self.get_grid()
        else:
            self.r = r
        if f is None:
            f = self.get_beam(r)
        else:
            self.f = f
        if n_init is None:
            n = self.analytic(r, f)
        else:
            n = n_init
        if self.__backend == 'cpu':
            func = self._numeric
        elif self.__backend == 'gpu':
            func = self._numeric_gpu
        if self.D != 0 or self.p_o != 0:
            n = func(f, eps, n, **kwargs)
        self._interp = None
        return n

    def _numeric(self, f, eps=1e-8, n_init=None, progress=False):
        def local_dict(n, f, n_D):
            return self._local_dict | dict(n=n, f=f, n_D=n_D)
        n_iters = int(1e9)
        n = np.copy(n_init)
        n_check = np.copy(n_init)
        base_step = 100
        skip_step = base_step * 5
        skip = skip_step  # next planned accuracy check iteration
        prediction_step = skip_step * 3
        n_predictions = 0
        norm = 1  # achieved accuracy
        norm_array = []
        iters = []
        if progress:
            t = tqdm(total=n_iters)
        else:
            t = None
        i = 0
        iter_jump = 1000
        while i < n_iters:
            step_iters = np.full((skip - i) // iter_jump + 1, iter_jump)
            step_iters[-1] = (skip - i) % iter_jump
            for step in step_iters:
                for j in range(step):
                    n_D = self.__diffusion(n)
                    ne.re_evaluate(self._numexpr_name, out=n, local_dict=local_dict(n, f, n_D))
                if t:
                    t.update(step)
                i = skip
            n_check[...] = n
            n_D = self.__diffusion(n)
            ne.re_evaluate('r_e', out=n, local_dict=local_dict(n, f, n_D))
            if self._validation_check(n):
                print(f'p_o: {self.p_o}, tau_r: {self.tau_r}')
                raise ValueError('Solution unstable!')
            norm = (np.linalg.norm(n[1:] - n_check[1:]) / np.linalg.norm(n[1:]))
            norm_array.append(norm)  # recording achieved accuracy for fitting
            iters.append(i)
            skip += skip_step
            if eps > norm:
                # print(f'Reached solution with an error of {norm:.3e}')
                break
            if i % prediction_step == 0:
                a, b = self.__fit_exponential(iters, norm_array)
                skip = int(
                    (np.log(eps) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
                if skip < 0:
                    raise ValueError('Instability in solution, solution convergance deviates from exponential behavior.')
                prediction_step = skip  # next prediction will be after another norm is calculated
                n_predictions += 1
                if t:
                    t.total = skip
                    t.refresh()
        self.n = n
        return n

    def _numeric_gpu(self, f_init, eps=1e-8, n_init=None, progress=False, ctx=None, queue=None):
        # Padding array to fit a group size
        N = n_init.shape[0]
        local_size = (256,)
        # global_size = (N % local_size[0] + N // local_size[0],)
        n = np.pad(np.copy(n_init), (0, local_size[0] - N % local_size[0]), 'constant', constant_values=(0, 0))
        n_check = np.copy(n)
        f = np.pad(f_init, (0, local_size[0] - N % local_size[0]), 'constant', constant_values=(0, 0))
        type_dev = np.float64
        n_f = n.astype(type_dev)
        n_check_f = n_check.astype(type_dev)

        base_step = self._gpu_base_step()
        skip_step = base_step * 5 + 1
        skip = skip_step  # next planned accuracy check iteration
        prediction_step = skip_step * 3
        n_predictions = 0
        norm_array = [1]
        iters = [1]
        # Getting equation constants
        # Increasing magnitude of the calculated value to increase floating point calculations
        unit = 1
        # n0, F, sigma, D, step = self.scale_parameters_units(unit)
        # f *= unit ** 2
        # eps *= unit / 10
        # self.analytic(r, f, n0=n0, F=F, sigma=sigma, out=n)
        if ctx is None or queue is None:
            ctx, prog, queue = cl_boilerplate()
        prog = self._configure_kernel(ctx, local_size, n.size)
        n_dev = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=n.astype(type_dev).nbytes)
        n_dev1 = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=n.astype(type_dev).nbytes)
        f_dev = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=f.astype(type_dev).nbytes)

        cl.enqueue_copy(queue, n_dev, n.astype(type_dev))
        cl.enqueue_copy(queue, n_dev1, n.astype(type_dev))
        cl.enqueue_copy(queue, f_dev, f.astype(type_dev))

        n_iters = int(1e10)  # maximum allowed number of iterations per solution
        if progress:
            t = tqdm(total=n_iters)
        else:
            t = None
        warning = 0
        i = 0
        iter_jump = 200000
        while i < n_iters:
            step_iters = np.full((skip - i) // iter_jump + 1, iter_jump)
            step_iters[-1] = (skip - i) % iter_jump
            for step in step_iters:
                event1 = prog.stencil_rde(queue, n.shape, local_size, n_dev, f_dev, np.int32(step), np.int32(N), n_dev1)
                event1.wait()
                if t:
                    t.update(step)
            i = skip
            event3 = cl.enqueue_copy(queue, n_check_f, n_dev1)
            event3.wait()
            event2 = prog.stencil_rde(queue, n.shape, local_size, n_dev, f_dev, np.int32(1), np.int32(N), n_dev1)

            if self._validation_check(n):
                raise ValueError('Solution unstable!')

            event4 = cl.enqueue_copy(queue, n_f, n_dev1)
            event4.wait()
            norm = (np.linalg.norm(n_f[1:N] - n_check_f[1:N]) / np.linalg.norm(n_f[1:N]))
            if norm < norm_array[-1]:
                norm_array.append(norm)  # recording achieved accuracy for fitting
                iters.append(i)
            else:
                print(f'Accuracy decrease at iteration {i}: {norm}')
                prediction_step += skip_step
            skip += skip_step
            if eps > norm:
                # print(f'Reached solution with an error of {norm/unit:.3e}')
                break
            if i % prediction_step == 0:
                a, b = self.__fit_exponential(iters, norm_array)
                skip1 = int((np.log(eps) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
                if skip1 > n_iters:
                    raise IndexError('Reached maximum number of iterations!')
                if skip1 < 0:
                    warning += 1
                    skip *= 2
                    if warning > 2:
                        raise IndexError('Instability in solution, accuracy decrease trend.')
                prediction_step = skip1  # next prediction will be after another norm is calculated
                n_predictions += 1
                skip = skip1
                if t:
                    t.total = skip1
                    t.refresh()
        cl.enqueue_copy(queue, n_f, n_dev)
        self.n = n_f[:N] / unit ** 2
        return n

    def _validation_check(self, n):
        return n.max() > self.n0 or n.min() < 0

    def _configure_kernel(self, ctx, local_size, global_size):
        return cl.Program(ctx, reaction_diffusion_jit(self.s, self.F, self.n0, self.tau * 1e4, self.sigma * 1e4, self.D,
                                                      self.step, self.dt * 1e6, global_size, local_size[0])).build()

    def _gpu_base_step(self):
        return 10 * int(np.log(self.D) * np.log(self.f0))

    def __diffusion(self, n):
        return laplace_1d(n)

    def analytic(self, r, f=None, out=None):
        """
        Derive a steady state precursor coverage using an analytical solution (for D=0)
        :param r: radially symmetric grid
        :param f: electron flux
        :return: precursor coverage profile
        """
        s = self.s
        F = self.F
        n0 = self.n0
        tau = self.tau
        sigma = self.sigma
        if f is None:
            f = self.get_beam(r)
        t_eff = (s * F / n0 + 1 / tau + sigma * f) ** -1
        n = s * F * t_eff
        if out is not None:
            out[:] = n[:]
        else:
            self.n = n
        # print('Solved analytically')
        return n

    def __fit_exponential(self, x0, y0):
        """
        Fit data to an exponential equation y = a*exp(b*x)

        :param x0: x coordinates
        :param y0: y coordinates
        :return: ln(a), b
        """
        x = np.array(x0)
        y = np.array(y0)
        p = np.polyfit(x, np.log(y), 1)
        a = p[1]
        b = p[0]
        # returning ln(a) to directly solve for desired x
        return a, b

    def get_bonds(self):
        """
        Define boundaries for the grid based on beam settings.
        :return: float
        """
        if self.beam_type == 'super_gauss':
            n = self.order
        else:
            n = 1
        r_lim = math.fabs((math.log(self.f0 + 1, 8) / 3 + 1) * (self.st_dev * 3) / (1.7 + math.log(n)))
        return r_lim

    @property
    def R(self):
        """
        Calculate normalized growth rate profile from precursor coverage.

        :return:
        """
        if self.n is None:
            print('No precursor coverage available! Find a solution first.')
            return
        self._R = self.n * self.sigma * self.f / self.s / self.F
        return self._R

    @property
    def r_max(self, interpolate=True):
        """
        Get position of the growth speed maximum.
        Peaks are considered symmetric if they not at 0.
        :return:
        """
        if interpolate:
            if self._interp is None:
                self.interpolate('R')
        maxima, Maxima = get_peak(self.r, self.R, self._interp)
        delta = np.fabs(Maxima - self.R_0)
        if delta < 1e-4:
            r_max = 0
        else:
            r_max = np.fabs(maxima.max())
        return r_max

    @property
    def r_max_n(self):
        """
        Get position of the growth speed maximum.
        Peaks are considered symmetric if they not at 0.
        :return:
        """
        r_max = self.r_max
        r_max_n = 2 * r_max / self.fwhm
        return r_max_n

    @property
    def R_max(self, interpolate=True):
        """
        Get peak growth rate.
        :return:
        """
        if interpolate:
            if self._interp is None:
                self.interpolate('R')
        _, maxima = get_peak(self.r, self.R, self._interp)
        R_max = maxima.max()
        return R_max

    @property
    def R_0(self):
        """
        Get growth rate at the center of the beam.
        :return:
        """
        if self._interp is None:
            self.interpolate('R')
        R_0 = float(self._interp(0))
        return R_0

    @property
    def R_ind(self):
        """
        Relative indent of an indented deposit.
        Normalized by Maxinum height, thus this parameter max value is 1.
        :return:
        """
        R_center = self.R_0
        R_max = self.R_max
        R_ind = (R_max - R_center) / R_max
        R_ind = np.round(R_ind, 8)
        return R_ind

    @property
    def R_ind1(self):
        """
        Relative indent of an indented deposit.
        Normalized by indent height, thus this parameter max value is inf.
        :return:
        """
        R_center = self.R_0
        R_max = self.R_max
        R_ind = (R_max - R_center) / R_center
        R_ind = np.round(R_ind, 8)
        return R_ind

    @property
    def fwhm_d(self):
        fwhm_d = deposit_fwhm(self.r, self.R)
        return fwhm_d

    def scale_parameters_units(self, scale=100):
        """
        Calculate parameters in larger/smaller length units.
        :param scale: scaling factor
        :return: n0, F, sigma, D, step
        """
        n0 = self.n0 * scale ** 2
        F = self.F * scale ** 2
        D = self.D / scale ** 2
        sigma = self.sigma / scale ** 2
        step = self.step / scale
        return n0, F, sigma, D, step

    @property
    def backend(self):
        """
        A backend for the numerical solution. Can be cpu or gpu.
        :return:
        """
        return self.__backend

    @backend.setter
    def backend(self, val):
        if val == 'cpu':
            self.__backend = val
        elif val == 'gpu':
            self.__backend = val
        else:
            print('Unidentified backend, use \'cpu\' or \'gpu\'. Defaulting to \'cpu\'')
            self.__backend = 'cpu'

    def plot(self, var, dpi=150):
        if var not in ['R', 'n', 'f']:
            raise ValueError(f'{var} is not spatially resolved. Use R, n or f')
        if var == 'R':
            y_label = 'R/sFV'
        elif var == 'n':
            y_label = 'n, 1/nm^2'
        elif var == 'f':
            y_label = 'f, 1/nm^2/s'
        fig, ax = plt.subplots(dpi=dpi)
        x = self.r
        y = self.__getattribute__(var)
        if y is None:
            raise ValueError(f'The attribute \'{var}\' is not set.')
        line, = ax.plot(x, y)
        ax.set_xlabel('r, nm')
        ax.set_ylabel(y_label)
        plt.show()

    def interpolate(self, var):
        if var not in ['R', 'n']:
            raise ValueError(f'{var} is not interpolated. Use R or n')
        x = self.r
        y = self.__getattribute__(var)
        method = self.interpolation_method
        if method == 'cubic':
            sp1 = CubicSpline(x, y)
        elif method == 'pchip':
            sp1 = PchipInterpolator(x, y)
        elif method == 'akima':
            sp1 = Akima1DInterpolator(x, y)
        elif method == 'spline':
            sp1 = make_interp_spline(x, y, k=3)
        else:
            raise ArithmeticError('Interpolation method not recognized. Use cubic, pchip, akima or spline')
        self._interp = sp1
        return sp1

    def estimate_se_flux(self, ie, yld):
        """
        Calculate Secondary electron surface flux based on beam current and SE yield coefficient.
        :param ie: beam current, A
        :param yld: SE yield coefficient
        :return: center se flux for a gaussian beam
        """
        es = EFluxEstimator()
        es.st_dev = self.st_dev
        es.yld = yld
        es.ie = ie
        self.f0 = es.f0_se
        return self.f0

    def _local_var_defs(self):
        """
        Definitions of the variables.
        :return:
        """
        text = ''
        try:
            text += super()._local_var_defs()
        except AttributeError:
            pass
        text += ('\n Experiment result parameters:\n'
                 'All growth rate values are normalized by sFV\n'
                 'R: Growth rate profile.\n'
                 'r: Radially symmetric grid, nm.\n'
                 'f: Electron flux profile, 1/nm^2/s.\n'
                 'n: Precursor coverage profile, 1/nm^2.\n'
                 'fwhm_d: Width(FWHM) of the growth rate (deposit) profile, nm.\n'
                 'r_max: Position of the growth rate maximum, nm.\n'
                 'r_max_n: Position of the growth rate maximum, normalised.\n'
                 'R_max: Maximum growth rate.\n'
                 'R_0: Growth rate at the center of the beam.\n'
                 'R_ind: Relative indent of the growth rate profiles with two maximums with R_max as divisor.\n'
                 'R_ind1: Relative indent of the growth rate profiles with two maximums, with R_0 as divisor.\n'
                 )
        return text

    def __copy__(self):
        pr = Experiment1D()
        pr.n0 = self.n0
        pr.s = self.s
        pr.F = self.F
        pr.tau = self.tau
        pr.sigma = pr.sigma
        pr.f0 = self.f0
        pr.D = self.D
        pr.step = self.step
        pr.fwhm = self.fwhm
        pr.V = self.V
        pr.beam_type = self.beam_type
        pr.order = self.order
        pr.backend = self.backend
        return pr


if __name__ == '__main__':
    pr = Experiment1D()
    pr.n0 = 2.8  # 1/nm^2
    pr.F = 1730.0  # 1/nm^2/s
    pr.s = 0.1
    pr.V = 0.05  # nm^3
    pr.tau = 500e-6  # s
    pr.D = 5e5  # nm^2/s
    pr.sigma = 0.022  # nm^2
    pr.fwhm = 500  # nm
    pr.f0 = 5e7
    pr.step = pr.fwhm // 200  # nm
    pr.beam_type = 'super_gauss'
    pr.order = 4
    pr.solve_steady_state(progress=True)
    pr.plot('R')
    pr.plot('f')
    pr.plot('n')
    print(pr.r_max / pr.fwhm)
