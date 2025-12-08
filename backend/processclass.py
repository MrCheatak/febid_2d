import math
import numpy as np
import numexpr_mod as ne
import pyopencl as cl
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, make_interp_spline

from backend.pycl_test import cl_boilerplate, reaction_diffusion_jit
from backend.dataclass import ContinuumModel
from backend.analyse import get_peak, deposit_fwhm
from backend.electron_flux import EFluxEstimator
from backend.diffusion import laplace_1d, laplacian_radial_1d


class Experiment1D(ContinuumModel):
    """
    Class representing a virtual experiment with precursor properties and beam settings.
    It allows calculation of a 2D precursor coverage and growth rate profiles based on the set conditions.
    """

    def __init__(self, backend='cpu', coords='radial'):
        super().__init__()
        self._r = None
        self.r = np.array([])
        self.f = None
        self.n = None
        self._R = None
        self._interp = None
        self.backend = backend
        self.interpolation_method = 'cubic'
        self._numexpr_name = 'r_e'
        self.coords = coords  # 'radial' or 'cartesian'
        self.equation = 'conventional'  # 'conventional' or 'dimensionless'
        self._tr = None
        n = 0.0
        n_D = 0.0
        f = 0.0
        local_dict = dict(s=self.s, F=self.F, n0=self.n0, tau=self.tau,
                          sigma=self.sigma, D=self.D, dt=self.dt, step=self.step)
        ne.cache_expression("(s*F*(1-n/n0) - n/tau - sigma*f*n + n_D*D)*dt + n", self._numexpr_name,
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
        if self.coords == 'cartesian':
            self.r = np.arange(-bonds, bonds, self.step)
        elif self.coords == 'radial':
            self.r = np.arange(0, bonds, self.step)
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
            result = func(f=f, eps=eps, n_init=n, **kwargs)
            n_all, timestamps = result
            n = n_all[-1]  # Use only the last solution
            self._tr = timestamps[-1]
        self._interp = None
        return n

    def solve_for_time(self, t, r=None, f=None, n_init=None, **kwargs):
        """
        Derive a precursor coverage for a given time.

        r, f and n_init must have the same length.
        If these parameters are not provided, they are generated automatically.

        If n_init is not provided, an analytical solution is used.

        :param t: time, s (or interval in microseconds if using interval mode)
        :param r: radially symmetric grid
        :param f: electron flux
        :param n_init: initial precursor coverage
        :return: precursor coverage profile or tuple of (solutions, timestamps) if interval is used
        """
        if r is None:
            r = self.get_grid()
        self.r = r
        if f is None:
            f = self.get_beam(r)
        self.f = f
        if n_init is None:
            n_init = np.zeros_like(r)
        elif n_init == "full":
            n_init = np.full_like(r, self.nr)
        self.n = n_init
        if type(t) in [float, int]:
            if t <= 0:
                return n_init
        if self.__backend == 'cpu':
            func = self._numeric
        elif self.__backend == 'gpu':
            func = self._numeric_gpu
        if self.D != 0 or self.p_o != 0:
            result = func(f, n_init=n_init, **kwargs)
            n_all, timestamps = result
            # Find the solution closest to time t
            if 'interval' in kwargs and kwargs['interval'] is not None:
                t_us = t * 1e6  # Convert to microseconds
                idx = np.searchsorted(timestamps, t_us)
                if idx == 0:
                    n = n_all[0]
                elif idx >= len(timestamps):
                    n = n_all[-1]
                else:
                    # Linear interpolation between the two closest solutions
                    t1, t2 = timestamps[idx - 1], timestamps[idx]
                    n1, n2 = n_all[idx - 1], n_all[idx]
                    weight = (t_us - t1) / (t2 - t1)
                    n = n1 + weight * (n2 - n1)
            else:
                n = n_all[-1]  # Use only the last solution
        self._interp = None
        return n

    def _numeric(self, f, eps=1e-8, n_init=None, interval=None, init_tol=1e-5, progress=False, verbose=False, plot_fit=False):
        """
        Solve the reaction-diffusion equation numerically.

        :param f: electron flux
        :param eps: solution accuracy
        :param n_init: initial precursor coverage
        :param interval: time interval between saved solutions in microseconds (if None, solve to steady state)
        :param progress: show progress bar
        :return: if interval is None: solution array; else: (solutions_list, timestamps_list)
        """
        def local_dict(n, f, n_D, base_loc_dict=None):
            if base_loc_dict is None:
                loc = self._local_dict
            else:
                loc = base_loc_dict.copy()
            loc = loc | dict(n=n, f=f, n_D=n_D)
            return loc
        def solve_step(n, f, time_elapsed, dt=None):
            n_D = self.__diffusion(n)
            # Use the reduced dt value in the local dict for numexpr evaluation
            loc_dict = base_local_dict | dict(n=n, f=f, n_D=n_D, dt=dt)
            ne.re_evaluate(self._numexpr_name, out=n, local_dict=loc_dict)
            if dt is not None:
                time_elapsed += dt
            else:
                time_elapsed += self.dt  # Increment time counter
            return time_elapsed
        base_local_dict = self._local_dict
        dt = self.dt
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
        time_elapsed = 0  # Initialize time counter
        fit_offset = 0
        # Storage for temporal solutions
        n_all = []
        timestamps = []
        next_save_time = 0
        interval_s = None

        if interval is not None:
            interval_s = interval * 1e-6  # Convert microseconds to seconds

        if progress:
            t = tqdm(total=n_iters)
        else:
            t = None
        i = 0
        iter_jump = 1000
        if interval is not None:
            n_all.append(np.copy(n))
            timestamps.append(time_elapsed)
            next_save_time += interval_s

        while i < n_iters:
            # Check if we need to save solution at this time

            step_iters = np.full((skip - i) // iter_jump + 1, iter_jump)
            step_iters[-1] = (skip - i) % iter_jump
            for step in step_iters:
                for j in range(step):
                    time_elapsed = solve_step(n, f, time_elapsed, dt)
                    if interval is not None and time_elapsed >= next_save_time:
                        n_all.append(np.copy(n))
                        timestamps.append(time_elapsed)
                        next_save_time += interval_s
                if t:
                    t.update(step)
                i = skip
            n_check[...] = n
            n_D = self.__diffusion(n)
            ne.re_evaluate(self._numexpr_name, out=n, local_dict=local_dict(n, f, n_D))
            if self._validation_check(n):
                print(f'p_o: {self.p_o}, tau_r: {self.tau_r}')
                raise ValueError('Solution unstable!')
            norm = (np.linalg.norm(n[1:] - n_check[1:]) / np.linalg.norm(n[1:])) # checking change in solution vector
            norm_array.append(norm)  # recording achieved accuracy for fitting
            iters.append(i)
            skip += skip_step
            if eps > norm:
                # print(f'Reached solution with an error of {norm:.3e}')
                break
            if i % prediction_step == 0:
                if len(norm_array) < 3:
                    continue
                a, b = self.__fit_exponential(iters[fit_offset:], norm_array[fit_offset:])
                if plot_fit:
                    if len(norm_array) < 4:
                        fig, ax = _plot_solution_covergence(iters, norm_array, a, b, eps, dt=dt)
                    else:
                        fig, ax = _plot_solution_covergence(iters, norm_array, a, b, eps, fig, ax, dt=dt)
                skip = int(
                    (np.log(eps) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
                if skip < 0:
                if len(norm_array) > 4:
                    fit_offset += 1  # moving fitting window forward
                if skip < 0 or skip > n_iters:
                    print(f'Predicted convergence step out of bounds: skip={skip}, i={i}, n_iters={n_iters}')
                    print(f'Fit parameters: a={a:.3e}, b={b:.3e}, eps={eps:.3e}')
                    print(f'Recent norms: {norm_array[-5:]}')
                    print(f'Recent iters: {iters[-5:]}')
                    raise ValueError('Instability in solution, solution convergence deviates from exponential behavior.')
                prediction_step = skip  # next prediction will be after another norm is calculated
                n_predictions += 1
                if t:
                    t.total = skip
                    t.refresh()

        # Interpolate the exact time when accuracy reaches eps
        if len(norm_array) >= 2 and norm < eps:
            # Fit exponential one last time to get accurate convergence time
            a, b = self.__fit_exponential(iters, norm_array)
            # Calculate the exact iteration where norm would equal eps
            iter_exact = (np.log(eps) - a) / b
            # Calculate the corrected time elapsed
            time_elapsed_corrected = iter_exact * dt
        else:
            time_elapsed_corrected = time_elapsed

        self.n = n
        if verbose:
            print(f'Time step: {self.dt} s')  # Print time step
            print(f'Total time elapsed (actual): {time_elapsed} s')  # Print actual time elapsed
            print(f'Total time elapsed (interpolated): {time_elapsed_corrected} s')  # Print interpolated time elapsed

        n_all.append(np.copy(n))
        timestamps.append(time_elapsed_corrected)
        return n_all, timestamps

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

        # Interpolate the exact time when accuracy reaches eps
        if len(norm_array) >= 2 and norm < eps:
            # Fit exponential one last time to get accurate convergence time
            a, b = self.__fit_exponential(iters, norm_array)
            # Calculate the exact iteration where norm would equal eps
            iter_exact = (np.log(eps) - a) / b
            # Calculate the corrected time elapsed
            time_elapsed_corrected = iter_exact * self.dt
        else:
            time_elapsed_corrected = i * self.dt

        cl.enqueue_copy(queue, n_f, n_dev)
        self.n = n_f[:N] / unit ** 2

        # Return in same format as CPU solver
        n_all = [n_f[:N] / unit ** 2]
        timestamps = [time_elapsed_corrected]
        return n_all, timestamps

    def _validation_check(self, n):
        return n.max() > self.n0 or n.min() < 0

    def _configure_kernel(self, ctx, local_size, global_size):
        return cl.Program(ctx, reaction_diffusion_jit(self.s, self.F, self.n0, self.tau * 1e4, self.sigma * 1e4, self.D,
                                                      self.step, self.dt * 1e6, global_size, local_size[0])).build()

    def _gpu_base_step(self):
        return 10 * int(np.log(self.D) * np.log(self.f0))

    def __diffusion(self, n):
        if self.coords == 'cartesian':
            n_D = laplace_1d(n) / self._step ** 2
        elif self.coords == 'radial':
            n_D = laplacian_radial_1d(n, self._r, self._step)
        else:
            raise ValueError('Coordinates not recognized, use \'cartesian\' or \'radial\'')
        return n_D

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
        r_lim = math.fabs((math.log(self.f0 + 1, 8) / 3 + 1) * (self.st_dev * 3) / (1.7 + math.log(n))) + 20
        if self.D:
            r_diff = math.sqrt(self.D * self.tau) * 5
        elif self.p_o:
            r_diff = math.sqrt(self.tau_r * self.p_o)
        return max(r_lim, r_diff)


    @property
    def tr(self, progress=False):
        """
        System relaxation time from a fully replenished state, s
        """
        r = self.get_grid()
        f = self.get_beam(r)
        n_init = np.full_like(r, self.nr)
        if self.__backend == 'cpu':
            func = self._numeric
        elif self.__backend == 'gpu':
            func = self._numeric_gpu
        # Solve with interval-based sampling
        n_solutions, timestamps = func(f=f, n_init=n_init, progress=progress)
        return timestamps[-1]

    @property
    def r(self):
        """
        Radially symmetric grid.
        :return:
        """
        return self._r

    @r.setter
    def r(self, val):
        self._r = val

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
        fig, ax = plt.subplots(dpi=dpi)
        x = self.r
        y = self.__getattribute__(var)
        if y is None:
            raise ValueError(f'The attribute \'{var}\' is not set.')
        if self.coords == 'cartesian':
            y = y[x >= 0]
            x = x[x >= 0]
        line, = ax.plot(x, y)
        if var == 'R':
            y_label = 'R/sFV'
            plt.title('Normalized growth rate profile')
        elif var == 'n':
            y_label = 'n, 1/nm^2'
            plt.title('Precursor coverage profile')
        elif var == 'f':
            y_label = 'f, 1/nm^2/s'
            plt.title('Electron flux profile')
        ax.set_ylabel(y_label)
        ax.set_xlabel('r, nm')
        plt.grid()

        # Add parameter box
        param_text = (
            f'Equation: {self.equation}\n'
            f'Symmetry: {self.coords}\n'
            f'Beam flux: {getattr(self, "f0", None):.3g}\n'
            f'Beam size: {getattr(self, "fwhm", None):.3g}\n'
            f'sF: {getattr(self, "s", None) * getattr(self, "F", None):.3g}\n'
            f'Res. time: {getattr(self, "tau", None):.3g}\n'
            f'D: {getattr(self, "D", None):.3g}\n'
            f'Diss. coef.: {getattr(self, "sigma", None):.3g}\n'
            f'Diff. repl.: {getattr(self, "p_o", None):.3g}\n'
            f'Depletion: {getattr(self, "tau_r", None):.3g}\n'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.5, param_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='center', bbox=props)

        plt.tight_layout()
        plt.show()

    def precursor_coverage_temporal(self, r, exposure=0.0, f=None, n_init=None, N=100, offset=10, **kwargs):
        """
        Derive a precursor coverage for given times with interpolation for denser point grid.

        r, f and n_init must have the same length.
        If f or n_init are not provided, they are generated automatically.

        If n_init is not provided, an analytical solution is used.

        :param r: radially symmetric grid
        :param exposure: total exposure time, s
        :param f: electron flux
        :param n_init: initial precursor coverage
        :param N: number of time points
        :param offset: number of points before/after exposure for offset
        :return: interpolated precursor coverage at center vs time, interpolated times
        """
        if r is None:
            r = self.get_grid()
        if f is None:
            f = self.get_beam(r)
        if n_init is None:
            n_init = np.zeros_like(r)
        elif n_init == "full":
            n_init = np.full_like(r, self.nr)

        # Calculate interval in microseconds to get approximately N points during exposure
        if exposure > 0:
            interval_us = (exposure / N) * 1e6  # Convert to microseconds
        else:
            interval_us = 1.0  # Default 1 microsecond interval
        if self.__backend == 'cpu':
            func = self._numeric
        elif self.__backend == 'gpu':
            func = self._numeric_gpu
        # Solve with interval-based sampling
        n_solutions, timestamps = func(f, n_init=n_init, interval=interval_us, **kwargs)

        # Extract center values from solutions
        n_arr = np.array(n_solutions)
        if self.coords == 'radial':
            n_center = n_arr[:, 0]
        elif self.coords == 'cartesian':
            n_center = n_arr[:, r.size // 2]
        else:
            # Default to radial if coords is not recognized
            n_center = n_arr[:, 0]

        timestamps = np.array(timestamps)

        # Create denser time grid for interpolation
        time_offset = exposure / offset if exposure > 0 else timestamps[-1] / offset
        times1 = np.linspace(-time_offset, 0, offset, endpoint=True)
        times2 = np.linspace(0, exposure if exposure > 0 else timestamps[-1], N)[1:]
        times3 = np.linspace(exposure if exposure > 0 else timestamps[-1],
                            (exposure if exposure > 0 else timestamps[-1]) + time_offset,
                            offset, endpoint=False)[1:]
        times_dense = np.concatenate((times1, times2, times3))

        # Create interpolation function using cubic spline
        from scipy.interpolate import CubicSpline
        interp_func = CubicSpline(timestamps, n_center)

        # Interpolate to dense grid
        n_all = np.zeros_like(times_dense)
        # Before exposure: use initial value
        n_all[times_dense < 0] = n_center[0]
        # During and after exposure: use interpolation
        mask = times_dense >= 0
        times_interp = times_dense[mask]
        # Clip to valid range
        times_interp = np.clip(times_interp, timestamps[0], timestamps[-1])
        n_all[mask] = interp_func(times_interp)

        return n_all, times_dense

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


def _plot_solution_covergence(iters, norm_array, a=None, b=None, eps=None, fig=None, ax=None, dt=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=150)
    else:
        ax.cla()
    ax.semilogy(iters, norm_array, 'o', label='Achieved accuracy')
    if a is not None and b is not None:
        x_fit = np.linspace(0, iters[-1], 300)
        y_fit = np.exp(a) * np.exp(b * x_fit)
        ax.semilogy(x_fit, y_fit, '-', label='Fitted convergence')
        if eps is not None:
            iter_exact = (np.log(eps) - a) / b
            ax.axvline(iter_exact, color='r', linestyle='--', label='Predicted convergence iteration')
            ax.semilogy([iter_exact], [eps], 'rx', label='Predicted convergence point')
    if dt is not None:
        try:
            # Add a secondary x-axis (top) showing time in seconds: time = iterations * dt
            ax_sec = ax.secondary_xaxis('top', functions=(lambda x: x * dt, lambda t: t / dt))
            ax_sec.set_xlabel('Time (s)')
        except Exception:
            # Fallback for older matplotlib: create a twinned top axis and set tick labels manually
            ax_top = ax.twiny()
            xticks = ax.get_xticks()
            ax_top.set_xticks(xticks)
            ax_top.set_xbound(ax.get_xbound())
            ax_top.set_xticklabels([f"{(val * dt):.3g}" for val in xticks])
            ax_top.set_xlabel('Time (s)')

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Norm')
        ax.set_title('Solution Convergence')
        ax.legend()
        ax.grid()
        fig.savefig('convergence_plot.png', dpi=300)
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        return fig, ax


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
