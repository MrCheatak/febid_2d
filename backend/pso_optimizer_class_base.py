import sys
import threading
import time
from copy import deepcopy

import numpy as np
import random
import matplotlib.pyplot as plt

from backend.processclass2 import Experiment1D_Dimensionless

from backend.logger import Logger

from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import  NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import animation

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed, wait

logger = Logger('pso_optimizer_results_last')


class Variable:
    name = None


class PSOVisualizer:
    def __init__(self, num_particles):
        self.fig, self.ax = plt.subplots()
        self.num_particles = num_particles
        self.scatter_plots = [None] * num_particles
        self.line_plots = [None] * num_particles
        self.update_flag = False
        self.positions = None
        self.alive = True

    def initialize(self, positions, xlim=None, ylim=None):
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        for i in range(self.num_particles):
            self.scatter_plots[i] = self.ax.scatter(*positions[i], marker='.')
            self.line_plots[i], = self.ax.plot(*positions[i], lw=1)

    def run(self):
        plt.draw()
        plt.pause(0.01)
        plt.show(block=False)
        while self.alive:
            if self.update_flag:
                self.update(self.positions)
                self.update_flag = False
                plt.pause(0.01)

    def update(self, positions):
        for i in range(self.num_particles):
            self.scatter_plots[i].set_offsets(positions[i])
            xdata, ydata = self.line_plots[i].get_data()
            xdata = np.append(xdata, positions[i, 0])
            ydata = np.append(ydata, positions[i, 1])
            self.line_plots[i].set_data(xdata, ydata)
        plt.draw()
        plt.pause(0.01)


class PSOVisualizerQT(QWidget):
    def __init__(self, num_particles, parent=None):
        super(PSOVisualizerQT, self).__init__(parent)
        self.num_particles = num_particles
        self.scatter_plots = [None] * num_particles
        self.line_plots = [None] * num_particles

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)


        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.update_flag = False
        self.positions = None
        self.alive = True

        self.show()

    def initialize(self, positions, xlim=None, ylim=None):
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        for i in range(self.num_particles):
            self.scatter_plots[i] = self.ax.scatter(*positions[i], marker='.')
            self.line_plots[i], = self.ax.plot(*positions[i], lw=0.5)
        self.canvas.draw()

    def update_viz(self, positions):
        if not self.alive:
            self.quit()
        for i in range(self.num_particles):
            xdata, ydata = self.line_plots[i].get_data()
            xdata = np.append(xdata, positions[i, 0])
            ydata = np.append(ydata, positions[i, 1])
            self.line_plots[i].set_data(xdata, ydata)
            self.scatter_plots[i].set_offsets(positions[i])
        self.canvas.draw()

    def save_animation(self, positions, filename):
        # Create the base figure
        fig, ax = plt.subplots()
        scatter = ax.scatter(*positions[0].T)
        lines = [ax.plot([], [], lw=0.5)[0] for _ in range(len(positions[0]))]  # Create a line plot for each particle

        # Update function for the animation
        def update(i):
            scatter.set_offsets(positions[i])
            for line, position in zip(lines, positions[i]):  # Update the line plot with new positions
                xdata, ydata = line.get_data()
                xdata = np.append(xdata, position[0])
                ydata = np.append(ydata, position[1])
                line.set_data(xdata, ydata)
            return scatter, *lines

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=range(len(positions)), blit=True)

        # Save the animation
        ani.save(filename, writer='pillow', fps=2)


def operator_func(ref_names, pr, param_names, param_vars):
    """
    Set parameters and run a simulation, process-safe
    :param ref_names:
    :param pr: experiment instance
    :param param_names: parameters names to set
    :param param_vars: parameters values to set
    :return:
    """
    # Set new parameters
    pr_stub = Experiment1D_Dimensionless()
    for name, var in zip(param_names, param_vars):
        pr.__setattr__(name, var)

    # Run simulation
    pr.solve_steady_state()

    R = pr.R
    r = pr.r

    # Analyze the result
    r_max = pr.r_max
    fwhm = pr.fwhm
    r_max_n = 2 * r_max / fwhm
    R_ind = pr.R_ind

    result = dict()
    for name in ref_names + param_names:
        result[name] = pr.__getattribute__(name)

    return result


class PSOptimizerThread(threading.Thread):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    def run(self):
        return self.optimizer.optimize_pso()


class PSOptimizerMP:
    """
    Particle swarm optimizer (PSO) with parallel multiprocessing.
    """
    best_solution = 0

    def __init__(self, num_of_variables=1, num_of_params=3, n_particles=100, n_iterations=1000, n_cores=None, progress_w=True):
        """

        :param num_of_variables:
        :param num_of_params:
        :param n_particles:
        :param n_iterations:
        """
        self._w = 0.7
        self._w_min = 0.4
        self._w_max = 0.9
        self._c1 = 1
        self._c2 = 2 - self._c1
        self.progressive_w = progress_w
        self._target_accuracy = 1e-8
        # Define reference parameters
        self.ref_vals = dict()
        # Define number of optimized variables
        self.var_names = []
        self.n_variables = num_of_variables
        # Define number of analyzed criteria
        self.n_params = num_of_params
        self.params = np.zeros(num_of_params)
        # Define the number of particles in the swarm
        # This is the number of random picks in the given area
        self.n_particles = n_particles
        if n_cores:
            self.core_workers = n_cores
        elif n_particles < cpu_count() - 4:
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
        self.positions = np.zeros((self.n_iterations+1, self.n_particles, num_of_variables))

        self._operator_func = None
        self._operator_base_args = None
        self.visualize = True
        self.visualizer = None
        self.optimizer_thread = PSOptimizerThread(self)
        # self.visualizer = PSOVisualizer(n_particles)
        # self.app = QApplication([])
        # self.visualizer = PSOVisualizerQT(n_particles)
        # self.visualizer.show()

    @property
    def target_accuracy(self):
        return self._target_accuracy

    @target_accuracy.setter
    def target_accuracy(self, val):
        self._target_accuracy = val

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, val):
        self._w = val

    @property
    def c1(self):
        return self._c1

    @c1.setter
    def c1(self, val):
        self._c1 = val

    @property
    def c2(self):
        return self._c2

    @c2.setter
    def c2(self, val):
        self._c2 = val

    def set_variables(self, *args):
        """
        Define optimized parameters. These parameters are going to be varied to find the best match based on the
        objective function and reference. Set min, max values, max increment and name vor each optimized variable.
        Name must correspond to one of the base parameters in the Experiment1D.

        :param args: list of tuples (min, max,max increment, name)
        :return:
        """
        if self.n_variables != len(args):
            raise ValueError('The number of specified variables does not match the number of optimized variables.')
        for i, arg in enumerate(args):
            self.position_limit[i] = arg[0], arg[1]
            self.velocity_limit[i] = arg[2]
            self.var_names.append(arg[3])

    def set_reference(self, *args):
        """
        Define reference parameters. The result of the simulation will be compared against these parameters.

        :param args: list of tuples (value, name)
        :return:
        """
        if self.n_params != len(args):
            raise ValueError(
                'The number of specified reference parameters does not match the number of analyzed criteria.')
        for i, arg in enumerate(args):
            self.ref_vals[arg[1]] = arg[0]

    def set_operator(self, operator_func):
        """
        Set the operator function that will be used to run the simulation.
        The operator function must return a dictionary with the same names as the reference parameters.

        :param operator_func: function
        :return:
        """
        self._operator_func = operator_func

    def configure_operator(self, *args):
        """
        Set the base arguments for the operator function.
        :param args:
        :return:
        """
        self._operator_base_args = args

    def objective_function(self, result):
        """
        Calls the minimised function and calculates the difference between the result and the reference.
        :return:
        """
        objective_value = 0
        for key in self.ref_vals.keys():
            try:
                objective_value += (result[key] - self.ref_vals[key]) ** 2
            except KeyError:
                raise KeyError(f'Parameter \'{key}\' not found in the result.')
        # Return the sum of the squared differences
        return objective_value

    def start_optimization(self):
        if self.visualize:
            self.app = QApplication(sys.argv)
            self.visualizer = PSOVisualizerQT(self.n_particles)
        self.optimizer_thread.start()
        if self.visualize:
            self.app.exec_()
        self.optimizer_thread.join()
        return self.global_best_position

    def optimize_pso(self, init_position=None):
        """
        Run the optimization.

        :param init_position:
        :return:
        """
        if self.core_workers > 1:
            executor = ProcessPoolExecutor(max_workers=self.core_workers)
        futures = []
        messages = []
        # Initialize the particle positions and velocities
        logger.info(
            f'Starting new optimization: N of particles: {self.n_particles}, N of iterations: {self.n_iterations}, target accuracy: {self.target_accuracy}')
        logger.info(f'Reference values: {[(name, val) for name, val in self.ref_vals.items()]}')
        logger.info(
            f'Optimized variables parameters (min, max, max_vel.): {[(name, pos, vel) for name, pos, vel in zip(self.var_names, self.position_limit, self.velocity_limit)]}')
        logger.info(f'Meta-parameters: w={self.w}, c1={self.c1}, c2={self.c2}, progressive_w={self.progressive_w}')
        logger.info('Initializing particle positions and velocities')
        for i in range(self.n_particles):
            for j in range(self.n_variables):
                if init_position:
                    self.particle_position[i, j] = init_position[j]
                else:
                    self.particle_position[i, j] = random.uniform(self.position_limit[j, 0], self.position_limit[j, 1])
                self.particle_velocity[i, j] = random.uniform(-self.velocity_limit[j], self.velocity_limit[j])
                self.local_best_position[i, :] = self.particle_position[i, :]
            if self.visualize:
                self.visualizer.initialize(self.particle_position, self.position_limit[0, 1], self.position_limit[1, 1])
            if self.core_workers > 1:
                f = executor.submit(self._operator_func, *self._operator_base_args, self.var_names,
                                    self.particle_position[i, :])
            else:
                f = self._operator_func(*self._operator_base_args, self.var_names, self.particle_position[i, :])
            futures.append(f)
        self.positions[0, :] = self.particle_position[:]
        if self.core_workers > 1:
            for i, f in enumerate(as_completed(futures)):
                result = f.result()
                self.local_best_value[i] = self.objective_function(result)
                # TODO: best solution is not equal to the particle position
                if self.local_best_value[i] < self.global_best_value:
                    self.global_best_value = self.local_best_value[i]
                    self.global_best_position[...] = self.particle_position[i, :]
                    self.best_solution = deepcopy(result)
        else:
            for i, f in enumerate(futures):
                result = f
                self.local_best_value[i] = self.objective_function(result)
                # TODO: best solution is not equal to the particle position
                if self.local_best_value[i] < self.global_best_value:
                    self.global_best_value = self.local_best_value[i]
                    self.global_best_position[...] = self.particle_position[i, :]
                    self.best_solution = deepcopy(result)
        # Define the particle swarm optimization parameters
        if self.progressive_w:
            rnd = 0.1
        w = self._w
        c1 = self._c1
        c2 = self._c2

        logger.info('Optimising...')
        logger.info(f'Initial best: Fit score: {self.global_best_value:.3f} '
                    f'Vals: {[(name, val) for name, val in zip(self.var_names, self.global_best_position)]}, '
                    f'Params: {[(name, val) for name, val in self.best_solution.items()]}')
        for t in range(self.n_iterations):
            logger.info(f'Iteration {t}')
            futures.clear()
            messages.clear()
            if self.progressive_w:
                rnd = 4 * rnd * (1 - rnd)
                w = self._w_min * rnd + (self._w_max - self._w_min) * t / self.n_iterations
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
                message = f'Trying new set {[(name, val) for name, val in zip(self.var_names, self.particle_position[i, :])]},'
                messages.append(message)
                if self.core_workers > 1:
                    future = executor.submit(self._operator_func, *self._operator_base_args, self.var_names,
                                        self.particle_position[i, :])
                else:
                    future = self._operator_func(*self._operator_base_args, self.var_names, self.particle_position[i, :])
                futures.append(future)
            if self.core_workers > 1:
                wait(futures)
            for i, f in enumerate(futures):
                if self.core_workers > 1:
                    result = f.result()
                else:
                    result = f
                current_value = self.objective_function(result)
                text = f'{messages[i]},  {[(name, result[name]) for name in self.ref_vals.keys()]}. Fit score: {current_value}'
                rating = (current_value - self.global_best_value) / self.local_best_value.max()
                logger.info(text, background_color=rating)
                if current_value < self.local_best_value[i]:
                    self.local_best_value[i] = current_value
                    self.local_best_position[i, :] = self.particle_position[i, :]
                # Check if current flock has reached a new global best position
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position[...] = self.local_best_position[i, :]
                    self.best_solution = deepcopy(result)
                    logger.info("New best value: Best fit = {}, Best parameters: {}".format(t, self.global_best_value,
                                                                                            self.global_best_position))
            if self.visualize:
                self.visualizer.positions = self.particle_position
                self.visualizer.update_flag = True
                self.visualizer.update_viz(self.particle_position)
            if self.global_best_value < self.target_accuracy:
                logger.info('Target accuracy reached.')
                break
            # Print the global best value at each iteration
            # print("Iteration {}: Best fit = {}, Best parameters: {}".format(t, self.global_best_value, self.global_best_position))
            self.positions[t+1, :] = self.particle_position[:]
        # Print the final global best position and value
        logger.info("Final global best position = {}".format(self.global_best_position))
        logger.info("Final global best value = {}".format(self.global_best_value))
        if self.visualize:
            self.app.quit()
        return self.global_best_position


if __name__ == '__main__':

    pr_d = Experiment1D_Dimensionless()
    pr_d.step = 5
    pr_d.tau_r = 100
    pr_d.p_o = 2
    pr_d.fwhm = 1400
    pr_d.f0 = 9e5
    pr_d.beam_type = 'super_gauss'
    pr_d.order = 4
    pr_d.backend = 'cpu'

    ns = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8]
    ns = np.array(ns)[::-1]

    logger.logger.setLevel('INFO')
    r_max_n_ref = [0.843, 0.893, 0.829]
    R_ind_ref = [0.894, 0.565, 0.333]
    fname = 'pso_results.txt'
    with open(fname, 'a') as f:
        f.write(f'{time.asctime()}\n')
        f.write('n\tr_max_n_opt\tR_ind_opt\ttau_r\tp_o\tFit score\n')

    for z in ns[:]:
        logger.info(f'Running for n={z}')
        pr_d.order = z
        for x, y in zip(r_max_n_ref, R_ind_ref):
            logger.info(f'Running for r_max_n={x}, R_ind={y}')
            pso = PSOptimizerMP(2, 2, n_particles=16, n_cores=16, n_iterations=100)
            pso.set_reference((x, 'r_max_n'), (y, 'R_ind'))
            pso.set_variables((0.1, 10, 0.2, 'p_o'), (1, 5000, 50, 'tau_r'))
            pso.set_operator(operator_func)
            pso.configure_operator(['r_max_n', 'R_ind'], pr_d)
            setattr(pso, '_sim', pr_d)
            result = pso.start_optimization()
            pso.visualizer.save_animation(pso.positions, f'pso_{z}_{x}_{y}.gif')
            fit_score = pso.global_best_value
            with open(fname, 'a') as f:
                f.write(f'{z}\t{x}\t{y}\t{result[0]}\t{result[1]}\t{fit_score}\n')
    pass