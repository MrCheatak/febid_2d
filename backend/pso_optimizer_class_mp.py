from copy import deepcopy

import numpy as np
import random
import matplotlib.pyplot as plt

from backend.processclass import Experiment1D
from backend.processclass2 import Experiment1D_Dimensionless
from backend.analyse import get_peak, deposit_fwhm

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed, wait


class Variable:
    name = None


def colored(text, t=None, b=None):
    """
    Color text based on the given scale.
    :param text: text to color
    :param t: text color scale
    :param b: background color scale
    :return:
    """
    if t is None:
        t = (0,0,0)
    if b is None:
        b = (255, 255, 255)
    return f"\033[48;2;{b[0]};{b[1]};{b[2]}m\033[38;2;{t[0]};{t[1]};{t[2]}m{text}\033[0m"


def green_red(val):
    """
    Define a green-red color scale.
    :param val:
    :return:
    """
    i = int(85 * val)
    if i < 0:
        i = 0
    if i > 85:
        i = 85
    r = i
    g = 85 - i
    return (120+r, 120+g, 120)


def run_sim(pr, param_name, param_vars):
    """
    Set parameters and run a simulation, process-safe
    :param pr: experiment instance
    :param param_name: parameters names to set
    :param param_vars: parameters values to set
    :return:
    """
    # Set new parameters
    for name, var in zip(param_name, param_vars):
        pr.__setattr__(name, var)

    # Run simulation
    pr.solve_steady_state()

    return pr


class PSO_OptimizerMP():
    """
    Particle swarm optimizer (PSO) with parallel multiprocessing.
    """
    R_max_ref = 0
    r_max_ref = 0
    fwhm_ref = 0
    best_solution = 0

    def __init__(self, pr: Experiment1D, num_of_variables=1, num_of_params=3, n_particles=100, n_iterations=1000):
        # Define number of optimized variables
        self.var_name = []
        self.n_variables = num_of_variables
        # Define number of analyzed criteria
        self.n_params = num_of_params
        self.params = np.zeros(num_of_params)
        # Define the number of particles in the swarm
        # This is the number of random picks in the given area
        self.n_particles = n_particles
        if n_particles < cpu_count() - 4:
            self.core_workers = n_particles
        else:
            self.core_workers = cpu_count() - 4
        # Define the number of iterations
        # This defines the number of 'migrations', each migration settles in a new area and probes it to find next optimal area
        self.n_iterations = n_iterations

        # Define the global best and local best positions and values
        self.global_best_position = np.zeros(num_of_variables)  # actual experimental parameters
        self.global_best_value = float('inf')  # convergence evaluation value
        self.local_best_position = np.zeros((self.n_particles, num_of_variables))
        self.local_best_value = np.zeros(self.n_particles)

        # Define the particle velocity and position limits
        self.velocity_limit = np.zeros(num_of_variables)
        self.position_limit = np.zeros((num_of_variables, 2))

        # Define the particle positions and velocities
        self.particle_position = np.zeros((self.n_particles, num_of_variables))
        self.particle_velocity = np.zeros((self.n_particles, num_of_variables))

        # Define solution engine
        self.pr = pr

    def set_variables(self, *args):
        """
        Define optimized parameters. Set min, max values, max increment and name.
        Name must correspond to one of the base parameters in the Experiment1D.

        :param args: list of tuples (min, max, dl, name)
        :return:
        """
        for i, arg in enumerate(args):
            self.position_limit[i] = arg[0], arg[1]
            self.velocity_limit[i] = arg[2]
            self.var_name.append(arg[3])

    # def set_reference(self, x, y, tau_r, p_o):

    #     self.r_ref = x
    #     self.R_ref = y
    #     max_x, max_y = get_peak(x, y)
    #     self.r_max_ref, self.R_max_ref = max_x.max(), max_y.mean()
    #     self.fwhm_ref = deposit_fwhm(x, y)
    #     self.tau_r_ref = tau_r
    #     self.p_o_ref = p_o
    def set_reference(self, fwhm, r_max, R_max):
        """
        Define reference data.
        :param x:
        :param y:
        :return:
        """
        self.r_max_ref = r_max
        self.fwhm_ref = fwhm
        self.R_max_ref = R_max

    def objective_function(self, pr:Experiment1D):
        """
        Objective function that calculates the difference between the result with current parameters
        and the reference.
        :param x: new parameters to run the simulation
        :return:
        """
        self.pr = pr

        R = self.pr.R
        r = self.pr.r

        # Analyze the result
        max_x, max_y = get_peak(r, R)
        r_max, R_max = max_x.max(), max_y.max()
        fwhm = deposit_fwhm(r, R)
        tau_r = self.pr.tau_r
        p_o = self.pr.p_o

        # Calculate the difference between the simulation and the experimental reference properties
        R_difference = (R_max - self.R_max_ref) / self.R_max_ref
        r_max_difference = (r_max - self.r_max_ref) / self.r_max_ref
        fwhm_difference = (fwhm - self.fwhm_ref) / self.fwhm_ref
        # tau_r_difference = (tau_r - self.tau_r_ref) / self.tau_r_ref
        # p_o_difference = (p_o - self.p_o_ref) / self.p_o_ref

        objective_value = 0
        objective_value += R_difference ** 2 + r_max_difference ** 2 + fwhm_difference ** 2
        # if self.global_best_value < 1e-2:
        # objective_value += tau_r_difference ** 2 + p_o_difference ** 2

        # Return the sum of the squared differences
        return objective_value

    def optimize_pso(self, init_position=None):
        """
        Run the optimization.

        :param init_position:
        :return:
        """
        executor = ProcessPoolExecutor(max_workers=self.core_workers)
        futures = []
        messages = []
        self.pr = run_sim(self.pr, self.var_name, (self.pr.sigma, self.pr.D, self.pr.tau))
        # Initialize the particle positions and velocities
        print('Initializing particle positions and velocities')
        for i in range(self.n_particles):
            for j in range(self.n_variables):
                if init_position:
                    self.particle_position[i, j] = init_position[j]
                else:
                    self.particle_position[i, j] = random.uniform(self.position_limit[j, 0], self.position_limit[j, 1])
                self.particle_velocity[i, j] = random.uniform(-self.velocity_limit[j], self.velocity_limit[j])
                self.local_best_position[i, :] = self.particle_position[i, :]
            # pr = run_sim(self.pr, self.var_names, self.particle_position[i, :])
            f = executor.submit(run_sim, self.pr, self.var_name, self.particle_position[i, :])
            futures.append(f)
        for i, f in enumerate(as_completed(futures)):
            pr = f.result()
            self.local_best_value[i] = self.objective_function(pr)
            if self.local_best_value[i] < self.global_best_value:
                self.global_best_value = self.local_best_value[i]
                self.global_best_position[...] = self.particle_position[i, :]
                self.best_solution = deepcopy(pr)
        # Define the particle swarm optimization parameters
        w = 0.7
        c1 = 1

        c2 = 2 - c1

        print('Optimising...')
        print(f'Initial best: Fit score: {self.global_best_value:.3f} {[(name, val) for name, val in zip(self.var_name, self.global_best_position)]}, '
              f'tau_r: {self.best_solution.tau_r}, p_o: {self.best_solution.p_o}.')
        self.plot_result(f'Initial, fit value:{self.global_best_value:.3f}')
        for t in range(self.n_iterations):
            print(f'Iteration {t}')
            futures.clear()
            messages.clear()
            for i in range(self.n_particles):
                # Update the particle velocity
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                for j in range(self.n_variables):
                    self.particle_velocity[i, j] = w * self.particle_velocity[i, j] + \
                                                   c1 * r1 * (self.local_best_position[i, j] -
                                                              self.particle_position[i, j]) + \
                                                   c2 * r2 * (self.global_best_position[j] -
                                                              self.particle_position[i, j])

                # Update the particle position
                # and get parameters for the next simulation
                new_position = np.empty(self.n_variables)
                for j in range(self.n_variables):
                    new_position[j] = self.particle_position[i, j] + self.particle_velocity[i, j]
                    if self.position_limit[j, 0] < new_position[j] < self.position_limit[j, 1]:
                        self.particle_position[i, j] = new_position[j]

                # Run simulation and check if the particle has reached a new local best position
                message = f'Trying new set {[(name, val) for name, val in zip(self.var_name, self.particle_position[i, :])]},'
                messages.append(message)
                future = executor.submit(run_sim, self.pr, self.var_name, self.particle_position[i,:])
                futures.append(future)
            wait(futures)
            for i, f in enumerate(futures):
                pr = f.result()
                current_value = self.objective_function(pr)
                text = f'{messages[i]},  tau_r: {pr.tau_r}, p_o: {pr.p_o}. Fit score: {current_value}'
                rating = (current_value - self.global_best_value)/self.local_best_value.max()
                color = green_red(rating)
                text_colored = colored(text, b=color)
                print(text_colored)
                if current_value < self.local_best_value[i]:
                    self.local_best_value[i] = current_value
                    self.local_best_position[i, :] = self.particle_position[i, :]

                # Check if current flock has reached a new global best position
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position[...] = self.local_best_position[i, :]
                    self.best_solution = deepcopy(pr)
                    self.pr = pr
                    print('New best value!')
                    print("Iteration {}: Best fit = {}, Best parameters: {}".format(t, self.global_best_value, self.global_best_position))
                    self.plot_result(f'Iteration {t} \nFit value: {current_value:.4f}')

            # Print the global best value at each iteration
            # print("Iteration {}: Best fit = {}, Best parameters: {}".format(t, self.global_best_value, self.global_best_position))


        # Print the final global best position and value
        print("Final global best position = {}".format(self.global_best_position))
        print("Final global best value = {}".format(self.global_best_value))
        return self.global_best_position

    def set_params(self, vals):
        for name, var in zip(self.var_name, vals):
            self.pr.__setattr__(name, var)

    def plot_result(self, text=None):
        fig, ax = plt.subplots()
        position = (0.02, 0.95)
        _ = ax.plot_from_exps(self.r_ref, self.R_ref, label='Reference')
        _ = ax.plot_from_exps(self.pr.r, self.pr.R, label='Solution')
        plt.text(*position, text, transform=ax.transAxes, fontsize=6, snap=True)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    pr_d = Experiment1D()
    pr_d.step = 1
    pr_d.tau = 20
    pr_d.D = 1e5
    pr_d.sigma = 1e-3
    pr_d.fwhm = 200
    pr_d.f0 = 1e7
    pr_d.beam_type = 'super_gauss'
    pr_d.order = 1

    pso = PSO_OptimizerMP(pr_d, 3, 3)
    pso.set_reference(1400, 0.8, 0.3)
    pso.set_variables((0.5, 2, 0.1, 'sigma'), (0.5, 2, 0.1, 'D'), (50, 200, 10, 'tau'))
    pso.optimize_pso()
