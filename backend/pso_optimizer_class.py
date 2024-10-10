import numpy as np
import random
import matplotlib.pyplot as plt

from backend.processclass import Experiment1D
from backend.analyse import get_peak, deposit_fwhm


class Variable:
    name = None


class PSO_Optimizer():
    r_ref = None
    R_ref = None
    R_max_ref = 0
    r_max_ref = 0
    fwhm_ref = 0
    tau_r_ref = 0
    p_o_ref = 0
    fig, ax = plt.subplots()

    def __init__(self, pr: Experiment1D, num_of_variables=1, num_of_params=3, n_particles=100, n_iterations=1000):
        # Define number of optimized variables
        self.n_variables = num_of_variables
        # Define number of analyzed criteria
        self.n_params = num_of_params
        # Define the number of particles in the swarm
        # This is the number of random picks in the given area
        self.n_particles = n_particles
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
        self.var_name = []

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

    def set_reference(self, fwhm, r_max, R_max):
        """
        Define reference data as a curve defined by (x,y) points.
        :param x:
        :param y:
        :return:
        """
        self.r_max_ref = r_max
        self.fwhm_ref = fwhm
        self.R_max_ref = R_max

    def objective_function(self, *x):
        """
        Objective function that calculates the difference between the result with current parameters
        and the reference.
        :param x: new parameters to run the simulation
        :return:
        """

        # Set new parameters
        for name, var in zip(self.var_name, x):
            self.pr.__setattr__(name, var)

        # Run simulation
        self.pr.solve_steady_state()
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
            self.local_best_value[i] = self.objective_function(*self.particle_position[i, :])  # Solution params
            if self.local_best_value[i] < self.global_best_value:
                self.global_best_value = self.local_best_value[i]
                self.global_best_position[...] = self.particle_position[i, :]

        # Define the particle swarm optimization parameters
        w = 0.7
        c1 = 1

        c2 = 2 - c1

        print('Optimising...')
        for t in range(self.n_iterations):
            for i in range(self.n_particles):
                # Update the particle velocity
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                for j in range(self.n_variables):
                    self.particle_velocity[i, j] = w * self.particle_velocity[i, j] + \
                                                   c1 * r1 * (self.local_best_position[i, j] - self.particle_position[
                        i, j]) + \
                                                   c2 * r2 * (self.global_best_position[j] - self.particle_position[
                        i, j])

                # Update the particle position
                # and get parameters for the next simulation
                new_position = np.empty(self.n_variables)
                for j in range(self.n_variables):
                    new_position[j] = self.particle_position[i, j] + self.particle_velocity[i, j]
                    if self.position_limit[j, 0] < new_position[j] < self.position_limit[j, 1]:
                        self.particle_position[i, j] = new_position[j]

                # Run simulation and check if the particle has reached a new local best position
                print(f'Trying new set {[(name, val) for name, val in zip(self.var_name, self.particle_position[i,:])]}, tau_r: {self.pr.tau_r}, p_o: {self.pr.p_o}. ', end='')
                current_value = self.objective_function(*self.particle_position[i, :])
                print(f'Fit score: {current_value}')
                # self.plot_result()
                if current_value < self.local_best_value[i]:
                    self.local_best_value[i] = current_value
                    self.local_best_position[i, :] = self.particle_position[i, :]

            # Check if current flock has reached a new global best position
            local_best = self.local_best_value.min()
            if local_best < self.global_best_value:
                self.global_best_value = local_best
                i = np.argmin(self.local_best_value)
                a,b,c = self.local_best_position[i, :]
                self.global_best_position[...] = a, b, c
                print('New best value!')
                self.objective_function(*self.global_best_position)
                self.plot_result(f'Iteration {t}')

            # Print the global best value at each iteration
            print("Iteration {}: Best objective value = {}, Best parameters: {}".format(t, self.global_best_value, self.global_best_position))


        # Print the final global best position and value
        print("Final global best position = {}".format(self.global_best_position))
        print("Final global best value = {}".format(self.global_best_value))
        return self.global_best_position

    def set_params(self, vals):
        for name, var in zip(self.var_name, vals):
            self.pr.__setattr__(name, var)

    def plot_result(self, text=None):
        fig, ax = plt.subplots()
        position = (0.02, 0.76)
        # _ = ax.plot(self.r_ref, self.R_ref, label='Reference')
        _ = ax.plot_from_exps(self.pr.r, self.pr.R, label='Solution')
        plt.text(*position, text, transform=ax.transAxes, fontsize=6, snap=True)
        plt.legend()
        plt.show()
