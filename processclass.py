import math
import numpy as np
import numexpr_mod as ne
import pyopencl as cl
from tqdm import tqdm

from pycl_test import cl_boilerplate
from dataclass import ContinuumModel


class Experiment2D(ContinuumModel):
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
        self.f = self.f0 * np.exp(-r ** 2 / (2 * self.st_dev ** 2))
        return self.f

    def get_grid(self, bonds=None):
        if not bonds:
            bonds = self.get_bonds()
        self.r = np.arange(-bonds, bonds, self.step)
        return self.r

    def solve_steady_state(self, r=None, f=None, eps=1e-8, n_init=None):
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

    def __numeric_gpu(self, r, f, eps=1e-8, n_init=None):
        s_ = np.array([self.s])
        F_ = np.array([self.F])
        n0_ = np.array([self.n0])
        tau_ = np.array([self.tau])
        sigma_ = np.array([self.sigma])
        D_ = np.array([self.D])
        step_ = np.array([self.step])
        dt_ = np.array([self.dt])

        n_iters = int(1e9)
        N = n_init.shape[0]
        local_size = (64,)
        global_size = (N % local_size[0] + N // local_size[0],)
        n = np.pad(np.copy(n_init), (0, N % local_size[0]), 'constant', constant_values=(0, 0))
        n_check = np.copy(n)
        n_D = np.zeros_like(n)
        f = np.pad(f, (0, N % local_size[0]), 'constant', constant_values=(0, 0))

        base_step = 100
        skip_step = base_step * 5
        skip = skip_step  # next planned accuracy check iteration
        prediction_step = skip_step * 3
        n_predictions = 0
        norm = 1  # achieved accuracy
        norm_array = []
        iters = []

        context, prog, queue = cl_boilerplate()
        index = np.arange(1, n.shape[0] - 2, dtype=np.int32)
        n_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=n.nbytes)
        n_D_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=n.nbytes)
        f_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=f.nbytes)
        index_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=index.nbytes)

        cl.enqueue_copy(queue, n_dev, n)
        cl.enqueue_copy(queue, index_dev, index)
        cl.enqueue_copy(queue, f_dev, f)

        for i in tqdm(range(n_iters)):
            if i % skip == 0 and i != 0:  # skipping array copy
                # event1.wait()
                event2.wait()
                cl.enqueue_copy(queue, n_check, n_dev)

            # event1 = prog[1].stencil_3(queue, global_size, local_size, np.int32(n.shape[0]), n_dev, n_D_dev)
            # event1 = prog[1].stencil(queue, global_size, local_size, n_dev, index_dev, n_D_dev)
            event2 = prog[1].reaction_equation(queue, n.shape, None, n_dev, s_, F_, n0_, tau_, sigma_, f_dev, D_,
                                               n_D_dev, step_,
                                               dt_)
            cl.enqueue_copy(queue, n, n_dev)

            if n.max() > self.n0 or n.min() < 0:
                raise ValueError('Solution unstable!')

            if i % skip == 0 and i != 0:  # skipping achieved accuracy evaluation
                cl.enqueue_copy(queue, n, n_dev)
                norm = (np.linalg.norm(n[1:] - n_check[1:]) / np.linalg.norm(n[1:]))
                norm_array.append(norm)  # recording achieved accuracy for fitting
                iters.append(i)
                skip += skip_step
            if eps > norm:
                print(f'Reached solution with an error of {norm:.3e}')
                break
            if i % prediction_step == 0 and i != 0:
                a, b = self.__fit_exponential(iters, norm_array)
                skip = int(
                    (np.log(eps) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
                prediction_step = skip  # next prediction will be after another norm is calculated
                n_predictions += 1

        cl.enqueue_copy(queue, n, n_dev)
        self.n = n
        return n

    def __diffusion(self, n):
        n_out = np.copy(n)
        n_out[0] = 0
        n_out[-1] = 0
        n_out[1:-1] *= -2
        n_out[1:-1] += n[2:]
        n_out[1:-1] += n[:-2]
        return n_out

    def analytic(self, r, f=None):
        if f is None:
            f = self.get_gauss(r)
        t_eff = (self.s * self.F / self.n0 + 1 / self.tau + self.sigma * f) ** -1
        self.n = n = self.s * self.F * t_eff
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
        r_lim = math.fabs((math.log(self.f0, 8) / 3.5) * (self.st_dev * 3))
        return r_lim

    @property
    def R(self):
        self._R = self.n * self.sigma * self.f / self.s / self.F
        return self._R

    @property
    def backend(self):
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
