from copy import deepcopy
from math import nan

import sympy.core
from sympy import Symbol, exp, log, lambdify, Abs
from scipy.optimize import basinhopping


def print_fun(x, f, accepted):
    print("at minimum %.4e accepted %d position %.4f" % (f, int(accepted), x))


class Model:
    """
    Model for the calculation of the process parameters (depleetion and diffusive replenishment)
    to precursor parameters using the FEBID Continuum model.

    See the source file for a usage example.
    """

    def __init__(self):
        self.T1 = Symbol('T1', real=True, positive=True)
        self.T2 = Symbol('T2', real=True, positive=True)
        self.T3 = Symbol('T3', real=True, positive=True)
        self.tau1 = Symbol('tau1', real=True, positive=True)
        self.tau2 = Symbol('tau2', real=True, positive=True)
        self.tau3 = Symbol('tau3', real=True, positive=True)
        self.rho1 = Symbol('rho1', real=True, positive=True)
        self.rho2 = Symbol('rho2', real=True, positive=True)
        self.rho3 = Symbol('rho3', real=True, positive=True)
        self.J = Symbol('J', real=True, positive=True)
        self.s = Symbol('s', real=True, positive=True)
        self.n0 = Symbol('n', real=True, positive=True)
        self.f0 = Symbol('f0', real=True, positive=True)
        self.FWHM = Symbol('FWHM', real=True, positive=True)
        self.kb = Symbol('kb', real=True, positive=True)
        self.Ea = Symbol('Ea', real=True, positive=True)
        self.k0 = Symbol('k0', real=True, positive=True)
        self.sigma = Symbol('sigma', real=True, positive=True)
        self.ED = Symbol('ED', real=True, positive=True)
        self.D0 = Symbol('D0', real=True, positive=True)
        self.Ea_expr = self.get_Ea_expr()
        self.k0_expr = self.get_k0_expr()
        self.sigma_expr = self.get_sigma_expr()
        self.ED_expr = self.get_ED_expr()
        self.D0_expr = self.get_D0_expr()
        self.subs_dict = {self.kb: 8.617e-5}
        self.model_vars = [self.J, self.s, self.n0, self.f0, self.FWHM]
        self.data_vars = [self.T1, self.T2, self.T3, self.tau1, self.tau2, self.tau3, self.rho1, self.rho2, self.rho3]
        self.result_vars = [self.Ea, self.k0, self.sigma, self.ED, self.D0]

        self.Ea_solution_bounds = (0.01, 5)
        self.Ea_solution_iterations = 5000
        self.minima_acceptance_score = 1e-11
        self.solution_debug = False

        self.conditions = {self.Ea: {'positive': True, 'bounds': self.Ea_solution_bounds},
                           self.k0: {'positive': True, 'bounds': (0, nan)},
                           self.sigma: {'positive': True, 'bounds': (0, 1)},
                           self.ED: {'positive': True, 'bounds': (0, 10)},
                           self.D0: {'positive': True, 'bounds': (0, nan)}
                           }

    def get_Ea_expr(self):
        kb = self.kb
        T1 = self.T1
        T2 = self.T2
        T3 = self.T3
        tau1 = self.tau1
        tau2 = self.tau2
        tau3 = self.tau3
        Ea = self.Ea
        Ea_expr = kb * T3 * log((exp(Ea * (T1 + 2 * T2) / (kb * T1 * T2)) * (tau1 - tau2) * (tau3 - 1)) / (
                exp(2 * Ea / (kb * T1)) * (tau2 - 1) * (tau1 - tau3) - exp(Ea * (T1 + T2) / (kb * T1 * T2)) * (
                tau1 - 1) * (tau2 - tau3))) - Ea
        return Ea_expr

    def get_k0_expr(self):
        kb = self.kb
        T1 = self.T1
        T2 = self.T2
        J = self.J
        s = self.s
        n0 = self.n0
        tau1 = self.tau1
        tau2 = self.tau2
        Ea = self.Ea
        k0_expr = -(J * s / n0) * (exp(Ea * (T1 + T2) / (kb * T1 * T2)) * (tau1 - tau2)) / (
                exp(Ea / (kb * T2)) * (tau1 - 1) - exp(Ea / (kb * T1)) * (tau2 - 1))
        return k0_expr

    def get_sigma_expr(self):
        kb = self.kb
        T1 = self.T1
        J = self.J
        s = self.s
        n0 = self.n0
        f0 = self.f0
        tau1 = self.tau1
        Ea = self.Ea
        k0 = self.k0
        sigma_expr = (exp(-Ea / (kb * T1)) * k0 + J * s / n0) * (tau1 - 1) / f0
        return sigma_expr

    def get_ED_expr(self):
        kb = self.kb
        T1 = self.T1
        T3 = self.T3
        J = self.J
        s = self.s
        n0 = self.n0
        rho1 = self.rho1
        rho3 = self.rho3
        Ea = self.Ea
        k0 = self.k0
        ED_expr = 1 / (T1 - T3) * (
                Ea * T1 - Ea * T3 + kb * T1 * T3 * log(k0 * n0 + exp(Ea / (kb * T1)) * J * s) - kb * T3 * T1 * log(
            k0 * n0 + exp(Ea / (kb * T3)) * J * s) + 2 * kb * T3 * T1 * log(rho1) - 2 * kb * T3 * T1 * log(rho3))
        return ED_expr

    def get_D0_expr(self):
        kb = self.kb
        T1 = self.T1
        J = self.J
        s = self.s
        n0 = self.n0
        rho1 = self.rho1
        Ea = self.Ea
        k0 = self.k0
        ED = self.ED
        FWHM = self.FWHM
        D0_expr = exp(-Ea / (kb * T1) + ED / (kb * T1)) * FWHM ** 2 * (
                k0 * n0 + exp(Ea / (kb * T1)) * J * s) * rho1 ** 2 / (4 * n0)
        return D0_expr

    def set_experiment_data(self, T1, T2, T3, tau1, tau2, tau3, rho1, rho2, rho3):
        self.subs_dict = {**self.subs_dict, self.T1: T1, self.T2: T2, self.T3: T3, self.tau1: tau1, self.tau2: tau2,
                          self.tau3: tau3, self.rho1: rho1, self.rho2: rho2, self.rho3: rho3}

    def set_params(self, J, s, n0, f0, FWHM):
        self.subs_dict = {**self.subs_dict, self.J: J, self.s: s, self.n0: n0, self.f0: f0, self.FWHM: FWHM}

    def solve_Ea(self, Ea_init=0.3):
        Ea_expr_abs = Abs(self.Ea_expr)
        Ea_expr_num = deepcopy(Ea_expr_abs)
        Ea_expr_num = Ea_expr_num.subs(self.subs_dict)
        func_np = lambdify(self.Ea, expr=Ea_expr_num, modules=['numpy'])
        solution = basinhopping(func_np, Ea_init,
                                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': [self.Ea_solution_bounds]}, T=5,
                                niter=self.Ea_solution_iterations, stepsize=0.5, disp=self.solution_debug,
                                callback=self.evaluate_minima)
        self.subs_dict[self.Ea] = sympy.core.Float(solution.x[0])
        return solution.x[0]

    def solve_k0(self):
        k0_expr_num = self.k0_expr.subs(self.subs_dict)
        self.subs_dict[self.k0] = k0_expr_num
        return k0_expr_num

    def solve_sigma(self):
        sigma_expr_num = self.sigma_expr.subs(self.subs_dict)
        self.subs_dict[self.sigma] = sigma_expr_num
        return sigma_expr_num

    def solve_ED(self):
        ED_expr_num = self.ED_expr.subs(self.subs_dict)
        self.subs_dict[self.ED] = ED_expr_num
        return ED_expr_num

    def solve_D0(self):
        D0_expr_num = self.D0_expr.subs(self.subs_dict)
        self.subs_dict[self.D0] = D0_expr_num
        return D0_expr_num

    def get_params(self, Ea_init=0.3, comments=False):
        """
        Calculate all parameters
        :param Ea_init: initial guess for activation energy
        :param comments: True will add comments to the result
        :return: dictionary with input data, model parameters, calculated parameters or a tuple with the same dictionary and comments
        """
        self.solve_Ea(Ea_init)
        self.solve_k0()
        self.solve_sigma()
        self.solve_ED()
        self.solve_D0()
        result = (deepcopy(self.subs_dict),)
        if comments:
            comment = ''
            for key in [self.Ea, self.k0, self.sigma, self.ED, self.D0]:
                if self.subs_dict[key] in [None, nan]:
                    comment += f'{key} failed to calculate\n '
                    continue
                if self.subs_dict[key].is_infinite:
                    comment += f'{key} is infinity\n '
                    continue
                if not self.subs_dict[key].is_extended_real:
                    comment += f'{key} is complex\n '
                    continue
                if self.subs_dict[key] == 0:
                    comment += f'{key} is zero\n '
                if float(self.subs_dict[key]) < 0:
                    comment += f'{key} is negative\n '
                if float(self.subs_dict[key]) > self.conditions[key]['bounds'][1]:
                    comment += f'{key} is too high\n '
                if float(self.subs_dict[key]) < self.conditions[key]['bounds'][0]:
                    comment += f'{key} is too low\n '
            # if self.subs_dict[self.Ea] < 0:
            #     comment += f'Ea is negative: {self.subs_dict[self.Ea]}\n '
            # if self.subs_dict[self.Ea] > 5:
            #     comment += f'Ea is too high: {self.subs_dict[self.Ea]}\n '
            # if self.subs_dict[self.k0] < 0:
            #     comment += f'k0 is negative: {self.subs_dict[self.k0]}\n '
            # if self.subs_dict[self.sigma] < 0:
            #     comment += f'sigma is negative: {self.subs_dict[self.sigma]}\n '
            # if self.subs_dict[self.sigma] > 1:
            #     comment += f'sigma is too high: {self.subs_dict[self.sigma]}\n '
            # if self.subs_dict[self.ED] < 0:
            #     comment += f'ED is negative: {self.subs_dict[self.ED]}\n '
            # if self.subs_dict[self.ED] > 5:
            #     comment += f'ED is too high: {self.subs_dict[self.ED]}\n '
            # if self.subs_dict[self.D0] < 0:
            #     comment += f'D0 is negative: {self.subs_dict[self.D0]}\n '
            result = (*result, comment)
        return result

    def lambdify_func(self, var, expr):
        expr_num = expr.subs(self.subs_dict)
        return lambdify(var, expr_num, modules=['numpy'])

    def evaluate_minima(self, x, f, accepted):
        if self.solution_debug:
            print("at minimum %.4e accepted %d position %.4f" % (f, int(accepted), x))
        if f < self.minima_acceptance_score:
            return True

    def plot_Ea(self):
        import matplotlib.pyplot as plt
        import numpy as np
        expr = Abs(self.Ea_expr)
        func_np = self.lambdify_func(self.Ea, expr)
        x = np.linspace(*self.Ea_solution_bounds, 100000) + 1e-13
        y = func_np(x)
        xy = np.vstack((x, y))
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()

    def flush_results(self):
        for key in self.result_vars:
            self.subs_dict.pop(key, None)

    def flush_experiment_data(self):
        for key in self.data_vars:
            self.subs_dict.pop(key, None)

    def flush_model_params(self):
        for key in self.model_vars:
            self.subs_dict.pop(key, None)


def keys_to_string(dictionary):
    return {str(key): value for key, value in dictionary.items()}


if __name__ == '__main__':
    print('Model test')
    # Input experimental data. It should contain three temperatures(T) and three of each process parameters
    # depletion(tau) and diffusive replenishment(rho) corresponding to those temperatures.
    data = {'T1': 283, 'T2': 298, 'T3': 303, 'tau1': 1100, 'tau2': 290, 'tau3': 38, 'rho1': 1.37, 'rho2': 0.25,
            'rho3': 0.18}
    print(f'Experiment data: {data}')
    # Model parameters. J is precursor flux, s is sticking coefficient, n0 is max. precursor coverage,
    # f0 is the electron flux at the beam center, FWHM is the beam width.
    model_params = {'J': 5400, 's': 0.0004, 'n0': 2.7, 'f0': 9e5, 'FWHM': 1400}
    print(f'Model params: {model_params}')
    model = Model()
    # Firstly, the experimental data should be set.
    # It must be a dictionary with the same keys as the set_experiment_data method() arguments.
    model.set_experiment_data(**data)
    # Then, the model parameters should be set.
    # It must be a dictionary with the same keys as the set_params method() arguments.
    model.set_params(**model_params)
    # The results are returned immediately from the get_params() method.
    # Optionally, the comments can be added to the result, containing errors and inconsistencies in the result.
    # The dictionary with the results returned also contains the input data.
    # Key bindings are the following:
    # 'T1', 'T2', 'T3', 'tau1', 'tau2', 'tau3', 'rho1', 'rho2', 'rho3' - input data
    # 'J', 's', 'n0', 'f0', 'FWHM' - model parameters
    # 'Ea', 'k0', 'sigma', 'ED', 'D0' - calculated parameters
    # Ea is the adsorption activation energy,
    # k0 is the desorption attempt frequency,
    # sigma is the dissociation cross-section,
    # ED is the diffusion activation energy,
    # D0 is the diffusion pre-exponential factor.
    result, comments = (model.get_params(comments=True))
    print(f'Result: {result}')
    print(f'Comments: {comments}')
