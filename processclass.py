import math
from pickle import dump
import numpy as np
import numexpr_mod as ne
import pyopencl as cl
from tqdm import tqdm
import matplotlib.pyplot as plt

from pycl_test import cl_boilerplate, reaction_diffusion_jit
from dataclass import ContinuumModel


class Experiment2D(ContinuumModel):
    """
    Class representing a virtual experiment with precursor properties and beam settings.
    It allows calculation of a 2D precursor coverage and growth rate profiles based on the set conditions.
    """
    r = None
    f = None
    n = None
    _R = None

    def __init__(self, backend='cpu'):
        super().__init__()
        self.backend = backend
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

    def get_gauss(self, r):
        """
        Generate electron beam profile based on a Gaussian.
        :param r: radially symmetric grid
        :return:
        """
        self.f = self.f0 * np.exp(-r ** 2 / (2 * self.st_dev ** 2))
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

    def solve_steady_state(self, r=None, f=None, eps=1e-8, n_init=None):
        """
        Derive a steady state precursor coverage.

        r, f and n_init must have the same length.
        If these parameters are not provided, they are generated automatically.

        If n_init is not provided, an analytical solution is used.

        eps should be changed together with step class attribute, otherwise the solution may fall through.

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
            f = self.get_gauss(r)
        else:
            self.f = f
        if n_init is None:
            n = self.analytic(r, f)
        else:
            n = n_init
        if self.__backend == 'cpu':
            func = self.__numeric
        elif self.__backend == 'gpu':
            func = self.__numeric_gpu
        if self.D != 0:
            n = func(r, f, eps, n)
        return n

    def __numeric(self, r, f=None, eps=1e-8, n_init=None):
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
        no_len = (i for i in range(1000000))
        # for i in tqdm(range(10000000), total=float("inf")):
        for i in range(100000000):
            if i % skip == 0 and i != 0:  # skipping array copy
                n_check[...] = n
            n_D = self.__diffusion(n)
            ne.re_evaluate('r_e', out=n, local_dict=self._local_dict)
            if n.max() > self.n0 or n.min() < 0:
                print(f'D: {self.D}, f0: {self.f0}')
                raise ValueError('Solution unstable!')
            if i % skip == 0 and i != 0:  # skipping achieved accuracy evaluation
                norm = (np.linalg.norm(n[1:] - n_check[1:]) / np.linalg.norm(n[1:]))
                norm_array.append(norm)  # recording achieved accuracy for fitting
                iters.append(i)
                skip += skip_step
            if eps > norm:
                # print(f'Reached solution with an error of {norm:.3e}')
                break
            if i % prediction_step == 0 and i != 0:
                a, b = self.__fit_exponential(iters, norm_array)
                skip = int(
                    (np.log(eps) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
                prediction_step = skip  # next prediction will be after another norm is calculated
                n_predictions += 1
        self.n = n
        return n

    def __numeric_gpu(self, r_init, f_init, eps=1e-8, n_init=None):
        n_iters = int(1e9)
        N = n_init.shape[0]
        local_size = (256,)
        # global_size = (N % local_size[0] + N // local_size[0],)
        n = np.pad(np.copy(n_init), (0, local_size[0] - N % local_size[0]), 'constant', constant_values=(0, 0))
        n_check = np.copy(n)
        n_D = np.zeros_like(n)
        f = np.pad(f_init, (0, local_size[0] - N % local_size[0]), 'constant', constant_values=(0, 0))
        r = np.pad(r_init, (0, local_size[0] - N % local_size[0]), 'constant', constant_values=(0, 0))
        type_dev = np.float64
        n_D_f = n_D.astype(type_dev)
        n_f = n.astype(type_dev)
        n_check_f = n_check.astype(type_dev)

        base_step = 50000
        skip_step = base_step * 5
        skip = skip_step  # next planned accuracy check iteration
        step = 0
        prediction_step = skip_step * 3
        n_predictions = 0
        norm = 1  # achieved accuracy
        norm_array = []
        iters = []
        # Getting equation constants
        s = self.s
        F = self.F
        n0 = self.n0
        tau = self.tau
        sigma = self.sigma
        D = self.D
        step = self.step
        dt = self.dt
        # Increasing magnitude of the calculated value to increase floating point calculations
        # Going from nm to 200 nm unit
        unit = 100
        n0, F, sigma, D, step = self.scale_parameters_units(unit)
        f *= unit**2
        eps *= unit/10
        self.analytic(r, f, n0=n0, F=F, sigma=sigma, out=n)
        context, prog, queue = cl_boilerplate()
        prog = cl.Program(context, reaction_diffusion_jit(s, F, n0, tau*1e4, sigma*1e4, D, step, dt*1e6, n.size, local_size[0])).build()
        # index = np.arange(1, n.shape[0] - 2, dtype=np.int32)
        n_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=n.astype(type_dev).nbytes)
        n_D_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=n.astype(type_dev).nbytes)
        f_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=f.astype(type_dev).nbytes)
        # index_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=index.nbytes)

        cl.enqueue_copy(queue, n_dev, n.astype(type_dev))
        # cl.enqueue_copy(queue, index_dev, index)
        cl.enqueue_copy(queue, f_dev, f.astype(type_dev))
        # t = tqdm(total=n_iters)
        i = 0
        # for i in tqdm(range(n_iters)):
        # for i in range(n_iters):
        iter_jump = 100000
        while i<n_iters:
            for q in range((skip-i) // iter_jump):
                step = iter_jump
                event1 = prog.stencil_rde(queue, n.shape, local_size, n_dev, f_dev, np.int32(step), np.int32(N))
                event1.wait()
                # t.update(step)
            else:
                step = (skip-i) % iter_jump
                event1 = prog.stencil_rde(queue, n.shape, local_size, n_dev, f_dev, np.int32(step), np.int32(N))
                event1.wait()
                # t.update(step)
            i = skip
            if i % skip == 0 and i != 0:  # skipping array copy
                # t.update(skip)
                # event.wait()
                event1.wait()
                # event2.wait()
                event3 = cl.enqueue_copy(queue, n_check_f, n_dev)
                event3.wait()
            event2 = prog.stencil_rde(queue, n.shape, local_size, n_dev, f_dev, np.int32(1), np.int32(N))
            # event1 = prog[1].stencil_3(queue, global_size, local_size, np.int32(n.shape[0]), n_dev, n_D_dev)
            # event1 = prog[1].stencil(queue, global_size, local_size, n_dev, index_dev, n_D_dev)
            # event2 = prog[1].reaction_equation(queue, n.shape, None, n_dev, s_, F_, n0_, tau_, sigma_, f_dev, D_,
            #                                    n_D_dev, step_,  dt_)
            # event1 = prog.stencil_operator(queue, n.shape, local_size, n_dev, n_D_dev, np.int32(N))
            # cl.enqueue_copy(queue, n_D_f, n_D_dev)
            # event2 = prog.reaction_equation(queue, n.shape, local_size, n_dev, f_dev, n_D_dev, np.int32(N))
            # cl.enqueue_copy(queue, n_f, n_dev)

            # if n.max() > self.n0 or n.min() < 0:
            #     raise ValueError('Solution unstable!')

            if i % skip == 0 and i != 0:  # skipping achieved accuracy evaluation
                # event.wait()
                event1.wait()
                # event2.wait()
                event4 = cl.enqueue_copy(queue, n_f, n_dev)
                event4.wait()
                norm = (np.linalg.norm(n_f[1:N] - n_check_f[1:N]) / np.linalg.norm(n_f[1:N]))
                norm_array.append(norm)  # recording achieved accuracy for fitting
                iters.append(i)
                skip += skip_step
            if eps > norm:
                # print(f'Reached solution with an error of {norm/unit:.3e}')
                break
            if i % prediction_step == 0 and i != 0:
                # event.wait()
                event1.wait()
                # event2.wait()
                a, b = self.__fit_exponential(iters, norm_array)
                skip = int((np.log(eps) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
                if skip > n_iters:
                    raise IndexError('Reached maximum number of iterations!')
                prediction_step = skip  # next prediction will be after another norm is calculated
                n_predictions += 1
                # t.total = skip
                # t.refresh()
        cl.enqueue_copy(queue, n_f, n_dev)
        self.n = n_f[:N] / unit**2
        return n

    def __diffusion(self, n):
        n_out = np.copy(n)
        n_out[0] = 0
        n_out[-1] = 0
        n_out[1:-1] *= -2
        n_out[1:-1] += n[2:]
        n_out[1:-1] += n[:-2]
        return n_out

    def analytic(self, r, f=None, s=None, F=None, n0=None, tau=None, sigma=None, out=None):
        """
        Derive a steady state precursor coverage using an analytical solution (for D=0)
        :param r: radially symmetric grid
        :param f: electron flux
        :return: precursor coverage profile
        """
        if s is None:
            s = self.s
        if F is None:
            F = self.F
        if n0 is None:
            n0 = self.n0
        if tau is None:
            tau = self.tau
        if sigma is None:
            sigma = self.sigma
        if f is None:
            f = self.get_gauss(r)
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
        r_lim = math.fabs((math.log(self.f0, 8) / 3) * (self.st_dev * 3))
        return r_lim

    @property
    def R(self):
        """
        Calculate normalized growth rate.

        :return:
        """
        if self.n is None:
            print('No precursor coverage available! Find a solution first.')
            return
        self._R = self.n * self.sigma * self.f / self.s / self.F
        return self._R

    def scale_parameters_units(self, scale=100):
        """
        Calculate parameters in larger length units.
        :param scale: scaling factor
        :return: n0, F, sigma, D, step
        """
        n0 = self.n0 * scale**2
        F = self.F * scale**2
        D = self.D / scale**2
        sigma = self.sigma /scale**2
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

    def plot(self, var):
        if var not in ['R', 'n', 'f']:
            raise ValueError(f'{var} is not spatially resolved. Use R, n or f')
        fig,ax = plt.subplots()
        x = self.r
        y = self.__getattribute__(var)
        line, = ax.plot(x, y)
        plt.show()

    def save_to_file(self, filename):
        """
        Save experiment to a file.
        :param filename: full file name (including path and extension)
        :return:
        """
        with open(filename, mode='wb') as f:
            dump(self, f)

