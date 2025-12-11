import math
import numpy as np
import numexpr_mod as ne
import pyopencl as cl
from tqdm import tqdm
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, make_interp_spline
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from backend.pycl_test import cl_boilerplate, reaction_diffusion_jit
from backend.dataclass import ContinuumModel
from backend.analyse import get_peak, deposit_fwhm
from backend.electron_flux import EFluxEstimator
from backend.diffusion import laplace_1d, laplacian_radial_1d, CrankNicolsonRadialSolver


class Experiment1D(ContinuumModel):
    """
    Class representing a virtual experiment with precursor properties and beam settings.
    It allows calculation of a 2D precursor coverage and growth rate profiles based on the set conditions.
    """

    def __init__(self, backend='cpu', coords='radial', num_scheme='fd'):
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
        self.dt_cn = 1.0  # placeholder for CN time step
        self.cn_dt_max_factor = 1000.0  # maximum factor to increase dt in CN scheme
        self._tr = None
        n = 0.0
        n_D = 0.0
        f = 0.0
        local_dict = dict(s=self.s, F=self.F, n0=self.n0, tau=self.tau,
                          sigma=self.sigma, D=self.D, dt=self.dt, step=self.step)
        ne.cache_expression("(s*F*(1-n/n0) - n/tau - sigma*f*n + n_D*D)*dt + n", self._numexpr_name,
                            local_dict=local_dict)

        # Numerical scheme for diffusion operator:
        # 'fd'  - explicit finite-difference Laplacian (legacy behaviour)
        # 'cn'  - radial Crank-Nicolson solver engine (implicit)
        self.num_scheme = num_scheme
        self.solver = None  # will hold CN solver if used

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

    def solve_steady_state(self, r=None, f=None, eps=1e-4, n_init=None, **kwargs):
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

    def solve_to_depletion(self, f, target_fraction=0.01, n_init=None, tol=1e-4, verbose=True, temporal=False, progress=False):
        """
        Solves time evolution until a specific depletion depth is reached.
        Supports temporal recording for visualization.
        """
        # 1. Setup Initial State
        if n_init is None:
            n = np.copy(n_init)  # FIX: self.n_init, not n_init
        else:
            n = np.copy(n_init)

        # 2. Configure Solver
        # Start with a safe, small dt to catch the "Cliff"
        dt = self.dt_fd
        solver = self.construct_cn_solver(f, dt)

        # 3. LOOK AHEAD: Calculate Exact Steady State
        if verbose: print("Pre-calculating steady state target...")
        n_ss = self.get_steady_state(solver)

        # 4. Calculate Distance
        def get_distance(n_current, n_target):
            return np.linalg.norm(n_current - n_target)

        dist_total = get_distance(n, n_ss)
        target_value = (n_init[0] - n_ss[0]) * target_fraction + n_ss[0]
        initial_value = n[0]

        # Handle edge case
        if dist_total < 1e-15:
            if verbose: print("System is already at steady state.")
            # Handle return signature consistency
            if temporal:
                return n, 0.0, n_ss, [n[0]], [0.0]
            else:
                return n, 0.0, n_ss

        # 5. Time Stepping Setup
        time_elapsed = 0.0

        # History for interpolation logic (Fraction vs Time)
        history_time = [0.0]
        history_fraction = [1.0]
        history_n_center = [n[0]]

        # Temporal recording (Center concentration vs Time)
        if temporal:
            n_all = [n[0]]
            time_all = [0.0]

        # Adaptive parameters
        n_iters = int(1e9)
        dt_min = self.dt_fd * 0.001
        dt_max = self.dt * self.cn_dt_max_factor

        if verbose:
            print(f"Total distance to cover: {dist_total:.4e}")
            print(f"Target fraction: {target_fraction:.4%}")

        # 6. Progress Bar Setup
        if progress:
            pbar = tqdm(total=target_value)
        else:
            pbar = None

        # 6. The Loop
        i = 0
        while i < n_iters:
            # Save state before attempting step (crucial for re-stepping)
            n_prev_step = np.copy(n)

            # --- Standard Adaptive Step ---
            n_full = solver.step(n, dt=dt)
            n_half = solver.step(n, dt=dt / 2)
            n_half_2 = solver.step(n_half, dt=dt / 2)

            # Error Estimate
            scale = np.maximum(np.abs(n_full), np.abs(n_half_2)) + 1e-20
            err = np.linalg.norm((n_full - n_half_2) / scale)

            if err < tol:
                # --- STEP CANDIDATE ACCEPTED ---

                # Check target using the CANDIDATE (n_half_2)
                # We have NOT updated 'n' or 'time_elapsed' yet
                dist_current = get_distance(n_half_2, n_ss)
                fraction_remaining = dist_current / dist_total
                current_value = n_half_2[0]

                if current_value <= target_value:
                    # --- TARGET REACHED ---

                    # 1. Interpolate Exact Time
                    # Previous point (valid history)
                    t_prev_log = history_time[-1]
                    frac_prev_log = history_fraction[-1]
                    n_c_prev = history_n_center[-1]

                    # Candidate point
                    t_curr_log = time_elapsed + dt
                    n_c = current_value

                    # Log-Linear Interpolation
                    y1, y2 = np.log(max(n_c_prev, 1e-20)), np.log(max(n_c, 1e-20))
                    target_log = np.log(target_value)

                    slope = (y2 - y1) / (t_curr_log - t_prev_log)

                    if abs(slope) > 1e-15:
                        time_exact = t_prev_log + (target_log - y1) / slope
                    else:
                        time_exact = t_curr_log

                    # 2. Re-step to Exact Time
                    # We use n_prev_step (state at time_elapsed)
                    dt_final = time_exact - time_elapsed

                    if dt_final > 1e-15:
                        if verbose: print(f"Performing final partial step: dt = {dt_final:.3e}s")
                        n_exact = solver.step(n_prev_step, dt=dt_final)
                    else:
                        n_exact = n_prev_step

                    if verbose:
                        print(f"Target reached.")
                        print(f"Time Exact ({target_fraction * 100:.1f}%): {time_exact:.4e} s")

                    # Update temporal lists one last time
                    if temporal:
                        n_all.append(n_exact[0])
                        time_all.append(time_exact)
                        return n_exact, time_exact, n_ss, np.array(n_all), np.array(time_all)
                    else:
                        return n_exact, time_exact, n_ss

                # --- CONTINUE SIMULATION ---
                # Update State
                n[:] = n_half_2
                time_elapsed += dt
                i += 1

                history_time.append(time_elapsed)
                history_fraction.append(fraction_remaining)
                history_n_center.append(n[0])

                if temporal:
                    n_all.append(n[0])
                    time_all.append(time_elapsed)

                if verbose and i % 500 == 0:
                    print(f"Time: {time_elapsed:.2e}s | Remaining: {fraction_remaining:.2%}")

                # Increase dt
                dt = min(dt * 1.5, dt_max)
                solver.set_dt(dt)

                if progress:
                    pbar.n = current_value
                    pbar.refresh()
            else:
                # Reject Step
                dt = max(dt * 0.5, dt_min)
                solver.set_dt(dt)

        # Fallback if max iters reached
        if temporal:
            return n, time_elapsed, n_ss, np.array(n_all), np.array(time_all)
        else:
            return n, time_elapsed, n_ss

    def _numeric(self, f, eps=1e-4, n_init=None, interval=None, init_tol=1e-5, depletion=0.01, progress=False, verbose=False, plot_fit=False):
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

        def reaction_step_exact(n, f, dt):
            """Exact solution for linear reaction ODE"""
            source = self.s * self.F
            decay_rate = self.s * self.F / self.n0 + 1 / self.tau + self.sigma * f

            # Exact solution: n(t) = n(0)*exp(-k*t) + S/k*(1 - exp(-k*t))
            exp_term = np.exp(-decay_rate * dt)
            n_new = n * exp_term + source / decay_rate * (1 - exp_term)

            # Handle zero decay rate (avoid division by zero)
            mask = decay_rate < 1e-15
            n_new[mask] = n[mask] + source * dt  # Limit case

            return n_new

        def solve_step(n, f, time_elapsed, dt=None):
            n_D = self.__diffusion(n)
            # Use the reduced dt value in the local dict for numexpr evaluation
            loc_dict = base_local_dict | dict(n=n, f=f, n_D=n_D, dt=dt)
            ne.re_evaluate(self._numexpr_name, out=n, local_dict=loc_dict)
            if dt is not None:
                time_elapsed += dt
            else:
                time_elapsed += self.dt
            return time_elapsed

        def solver_step_im(n, f, time_elapsed, dt=None):
            """
            Solve one time step using the coupled Crank-Nicolson solver.

            Now handles Diffusion + Reaction + Source in a single implicit step.
            No Strang splitting required.
            """
            if dt is None:
                dt = self.dt

            # The solver.step() method now handles the full physics (D, K, S).
            # It returns a new array, so we copy it back into 'n' to update state in-place.
            n_new = self.solver.step(n, dt=dt)
            n[:] = n_new

            # Update time
            if dt is not None:
                time_elapsed += dt
            else:
                time_elapsed += self.dt

            return time_elapsed
        # Choose solver function and time step based on numerical scheme
        if self.num_scheme == 'cn':
            r = self._r
            D = self.D
            # Pre-calculate Linear Reaction Terms
            # Decay rate K(r) = s*F/n0 + 1/tau + sigma*f
            # Source term S(r) = s*F
            # Note: We assume 'f' is constant for this solve run.
            reaction_k = self.s * self.F / self.n0 + 1 / self.tau + self.sigma * f
            reaction_s = self.s * self.F
            # dt = min(self.dt_des, self.dt_diss) # Crank-Nicolson allows larger time steps
            dt = self.dt_fd * 5
            dt_min = self.dt_fd * 0.1
            dt_max = self.dt * self.cn_dt_max_factor
            dts = []
            self.dt_cn = dt
            solver = self.construct_cn_solver(f, dt)
            solver_func = solver_step_im
            eps_pre1 = eps * 5
            eps_pre2 = eps * 1.5
            eps_final = eps
            eps_list = [eps_pre1, eps_pre2, eps_final]
        else:
            dt = self.dt
            solver_func = solve_step
            eps_list = [eps]
        base_local_dict = self._local_dict

        n_iters = int(1e9)  # maximum allowed number of iterations per solution
        n = np.copy(n_init)
        n_check = np.copy(n_init)
        n_half_2 = np.copy(n_init)
        tol = init_tol
        err_prev=tol
        base_step = 100
        skip_step = base_step * 5
        skip = skip_step  # next planned accuracy check iteration
        prediction_step = skip_step * 3
        n_predictions = 0
        norm = 1  # achieved accuracy
        norm_array = []
        iters = []
        time_array = []
        time_elapsed = 0  # Initialize time counter
        fit_offset = 0
        # Storage for temporal solutions
        n_all = []
        timestamps = []
        next_save_time = 0
        interval_s = None
        final_eps = False
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
        for k, ep in enumerate(eps_list):
            while i < n_iters:
                # Check if we need to save solution at this time

                step_iters = np.full((skip - i) // iter_jump + 1, iter_jump)
                step_iters[-1] = (skip - i) % iter_jump
                for step in step_iters:
                    for j in range(step):
                        time_elapsed = solver_func(n, f, time_elapsed, dt)
                        if interval is not None and time_elapsed >= next_save_time:
                            n_all.append(np.copy(n))
                            timestamps.append(time_elapsed)
                            next_save_time += interval_s
                    if t:
                        t.update(step)
                i = skip

                # Check convergence by comparing current state with saved state
                if self.num_scheme == 'cn':
                    # Use the exact operator from the class
                    res = self.solver.compute_residual(n)
                    # Metric: Relative L2 Norm of the residual (1/s)
                    # || dn/dt || / || n ||
                    norm = np.linalg.norm(res) / np.linalg.norm(n)
                    n_check[...] = n  # Save current state for next iteration
                    solver_step_im(n, f, time_elapsed, dt / 2)
                    solver_step_im(n, f, time_elapsed, dt / 2)
                    n_half_2[...] = n
                    n[...] = n_check  # Restore state
                    time_elapsed = solver_step_im(n, f, time_elapsed, dt)
                    # err = np.max(np.abs(n_half_2 - n))
                    err = np.linalg.norm(n_half_2[:] - n[:]) / np.linalg.norm(n_half_2[:])
                    if verbose and i % (skip_step * 10) == 0:
                            print(f"CN iter {i}: norm = {norm:.3e}, n_max = {n.max():.4f}, n_min = {n.min():.4f}")
                else:
                    # For explicit FD, take one full step and two half-steps to estimate error
                    n_check[...] = n  # Save current state for next iteration
                    # Two half-steps
                    n_D = self.__diffusion(n)
                    loc_dict = base_local_dict | dict(n=n, f=f, n_D=n_D)
                    loc_dict['dt'] = dt / 2
                    ne.re_evaluate(self._numexpr_name, out=n, local_dict=loc_dict)
                    n_D = self.__diffusion(n)
                    loc_dict = base_local_dict | dict(n=n, f=f, n_D=n_D)
                    loc_dict['dt'] = dt / 2
                    ne.re_evaluate(self._numexpr_name, out=n, local_dict=loc_dict)
                    n_half_2[...] = n  # Save two half-step result
                    n[...] = n_check  # Restore state
                    # One full step
                    n_D = self.__diffusion(n)
                    loc_dict = base_local_dict | dict(n=n, f=f, n_D=n_D)
                    ne.re_evaluate(self._numexpr_name, out=n, local_dict=loc_dict)
                    norm = np.linalg.norm(n[:] - n_check[:])/np.linalg.norm(n[:]) / dt  # relative norm
                    if self._validation_check(n):
                        print(f'p_o: {self.p_o}, tau_r: {self.tau_r}')
                        raise ValueError('Solution unstable!')

                norm_array.append(norm)  # recording achieved accuracy for fitting
                iters.append(i)
                time_array.append(time_elapsed)
                skip += skip_step
                if ep > norm:
                    # print(f'Reached solution with an error of {norm:.3e}')
                    fit_offset += 2
                    if k+1 == len(eps_list) - 1:  # final eps reached
                        final_eps = True
                        dt = self.dt_fd
                        dts.append(dt)
                        iters.append(i)
                        norm_array.append(norm)
                        time_array.append(time_elapsed)
                    if plot_fit:
                        a, b = self.__fit_exponential(iters[fit_offset:], norm_array[fit_offset:])
                        if len(norm_array) < 4:
                            fig, ax = _plot_solution_covergence(iters, norm_array, a, b, ep, dt=dt, times=time_array)
                        else:
                            fig, ax = _plot_solution_covergence(iters, norm_array, a, b, ep, fig, ax, dt=dt, times=time_array)
                    break
                if i % prediction_step == 0 or i > prediction_step:
                    if len(norm_array) < 3:
                        continue
                    a, b = self.__fit_exponential(iters[fit_offset:], norm_array[fit_offset:])
                    if plot_fit:
                        if len(norm_array) < 4:
                            fig, ax = _plot_solution_covergence(iters, norm_array, a, b, ep, dt=dt, times=time_array)
                        else:
                            fig, ax = _plot_solution_covergence(iters, norm_array, a, b, ep, fig, ax, dt=dt, times=time_array)
                    skip = int(
                        (np.log(ep) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
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
                if self.num_scheme == 'cn':
                    if not final_eps:
                        if err < tol:
                            err_prev = err
                            dt_new = dt * 0.9 * (tol / err) ** (1/3) * (tol / err_prev) ** 0.2  # adjust time step based on error
                            dt = np.clip(dt_new, dt_min, dt_max)
                            dts.append(dt)
                        else:  # Reject step
                            dt = dt * 0.9 * (tol / err) ** (0.35)  # Reduce dt
                            dt = max(dt, dt_min)
                    else:
                        dt = self.dt_fd
                        dts.append(dt)
                    self.dt_cn = dt

                    if t:
                        t.total = skip
                        t.refresh()

        # Interpolate the exact time when accuracy reaches eps
        if len(norm_array) >= 2 and norm < eps:
            if self.num_scheme == 'cn':
                # We want to find t where norm(t) = eps
                # Model: ln(norm) = slope * time + intercept

                # Use the last few points (e.g., last 5) for better local accuracy near steady state
                n_points = min(len(norm_array), 5)

                # Get arrays for fitting
                y_data = np.log(np.array(norm_array[-n_points:]))
                x_data = np.array(time_array[-n_points:])

                # Linear regression: y = mx + c
                # Using numpy's polyfit for stability (degree 1)
                slope, intercept = np.polyfit(x_data, y_data, 1)

                # Solve for t: ln(eps) = slope * t + intercept
                # t = (ln(eps) - intercept) / slope
                target_log = np.log(eps)
                time_elapsed_corrected = (target_log - intercept) / slope
            elif self.num_scheme == 'fd':
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
            if self.num_scheme == 'cn' and len(dts) > 0:
                print(f'Adaptive time stepping used: dt_min = {min(dts):.3e} s, dt_max = {max(dts):.3e} s, dt_final = {dts[-1]:.3e} s')
            else:
                print(f'Time step: {self.dt} s')  # Print time step
            print(f'Total time elapsed (actual): {time_elapsed} s')  # Print actual time elapsed
            print(f'Total time elapsed (corrected): {time_elapsed_corrected} s')  # Print corrected time elapsed
            print(f'Number of convergence checks: {len(iters)}')
            print(f'Final iteration: {i}, Final norm: {norm:.3e}')

        n_all.append(np.copy(n))
        timestamps.append(time_elapsed_corrected)
        return n_all, timestamps

    def construct_cn_solver(self, f, dt):
        if self._r is None:
            self.get_grid()
        r = self._r
        D = self.D
        # Pre-calculate Linear Reaction Terms
        # Decay rate K(r) = s*F/n0 + 1/tau + sigma*f
        # Source term S(r) = s*F
        # Note: We assume 'f' is constant for this solve run.
        reaction_k = self.s * self.F / self.n0 + 1 / self.tau + self.sigma * f
        reaction_s = self.s * self.F
        # dt = min(self.dt_des, self.dt_diss) # Crank-Nicolson allows larger time steps
        theta = 0.5  # Crank-Nicolson weighting
        stability_criterion = self.system_stiffness * self.dt_fd * 100
        if stability_criterion >= 2:
            # Stiff proble
            theta = 1.0  # Backward Euler
        # Initialize solver with reaction terms
        self.solver = CrankNicolsonRadialSolver(
            r=r,
            D=D,
            dt=dt,
            reaction_k=reaction_k,
            reaction_s=reaction_s,
            bc_outer="neumann",
            theta=theta
        )
        return self.solver

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

    def get_steady_state(self, solver=None):
        """
        Calculate steady state solution by solving the BVP:
        L*n = -S, where L = D*(d2/dr2 + 1/r d/dr) - K
        1. Build Tridiagonal Matrix for L
        2. Solve L*n = -S
        3. Return n_steady_bvp
        :param solver: CrankNicolsonRadialSolver instance (optional)
        :return: steady state precursor coverage profile
        """
        if solver is None:
            solver = self.construct_cn_solver(self.f, self.dt)
        # Build Tridiagonal Matrix for L = D*(d2/dr2 + 1/r d/dr) - K
        D = self.D
        N = solver.N
        r = solver.r
        dr = solver.dr
        K = solver.k if isinstance(solver.k, np.ndarray) else np.full(N, solver.k)
        S = solver.s if isinstance(solver.s, np.ndarray) else np.full(N, solver.s)
        diagonals = [np.zeros(N), np.zeros(N), np.zeros(N)]  # Lower, Main, Upper

        inv_dr2 = 1.0 / (dr ** 2)

        # Interior Points
        for i in range(1, N - 1):
            # Coeffs for: D * (n[i+1] - 2n[i] + n[i-1])/dr^2
            c2 = D * inv_dr2
            # Coeffs for: D * (1/r) * (n[i+1] - n[i-1])/(2dr)
            c1 = D / (2 * r[i] * dr)

            diagonals[0][i - 1] = c2 - c1  # Lower (coefficient of i-1)
            diagonals[1][i] = -2 * c2 - K[i]  # Main  (coefficient of i)
            diagonals[2][i + 1] = c2 + c1  # Upper (coefficient of i+1)

        # BC: r=0 (Symmetry) -> 4 * D * (n[1]-n[0])/dr^2 - K[0]*n[0]
        diagonals[1][0] = -4 * D * inv_dr2 - K[0]
        diagonals[2][1] = 4 * D * inv_dr2

        # BC: Outer (Neumann) -> 2 * D * (n[N-2]-n[N-1])/dr^2 - K[-1]*n[-1]
        diagonals[0][N - 2] = 2 * D * inv_dr2
        diagonals[1][N - 1] = -2 * D * inv_dr2 - K[-1]

        # Construct exact arrays for diags
        # Lower diagonal (offset -1): Entry k corresponds to A[k+1, k]
        # We stored A[i, i-1] at index i-1.
        # Letting k = i-1, we need index k. So we take the sequence from 0.
        lower_data = diagonals[0][:-1]  # Cut off the last unused element

        # Main diagonal (offset 0)
        main_data = diagonals[1]

        # Upper diagonal (offset +1): Entry k corresponds to A[k, k+1]
        # We stored A[i, i+1] at index i+1.
        # Letting k = i, we need index k+1. So we shift left by 1.
        upper_data = diagonals[2][1:]  # Cut off the first unused element

        # Construct sparse matrix
        matrix = diags(
            [lower_data, main_data, upper_data],
            [-1, 0, 1], shape=(N, N), format='csc'
        )

        # Solve L*n = -S
        rhs = -S
        n_steady_bvp = spsolve(matrix, rhs)
        return n_steady_bvp

    def __diffusion(self, n, coords=None, num_scheme=None):
        """Compute diffusion term n_D.

        For historical reasons this returned the spatial Laplacian times D
        via an explicit finite-difference scheme. To support a more stable
        implicit scheme, the behaviour is controlled by `self.num_scheme`:

        - 'fd': explicit finite-difference Laplacian (original behaviour)
        - 'cn': radial Crank-Nicolson solver used as diffusion engine
        """
        c = coords if coords is not None else self.coords
        ns = num_scheme if num_scheme is not None else self.num_scheme

        if c == 'cartesian':
            n_D = laplace_1d(n) / self._step ** 2
            return n_D
        elif c == 'radial':
            if ns == 'fd':
                n_D = laplacian_radial_1d(n, self._r, self._step)
                return n_D
            elif ns == 'cn':
                # Implicit Crank-Nicolson radial diffusion engine
                # This returns the result after a full CN step (not just the Laplacian)
                n_D = self.solver.step(n)
                return n_D
            else:
                raise ValueError(f"Unknown numerical scheme '{ns}', expected 'fd' or 'cn'")
        else:
            raise ValueError("Coordinates not recognized, use 'cartesian' or 'radial'")

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
    def decay_rate(self):
        """
        System mathematical decay rate, 1/s
        """
        k = self.s * self.F / self.n0 + 1 / self.tau + self.sigma * self.f.max()
        return k

    @property
    def system_stiffness(self):
        """
        System stiffness ratio
        """
        k = self.decay_rate
        stiffness = k + 4 * self.D / self.step ** 2
        return stiffness

    @property
    def dt_fd(self):
        """
        Time step limit for explicit finite-difference scheme, s
        """
        return self.dt

    # @property
    # def dt_cn(self):
    #     """
    #     Time step limit for implicit Crank-Nicolson scheme, s
    #     """
    #     return min(self.dt_des, self.dt_diss)

    @property
    def tr(self):
        """
        System relaxation time from a fully replenished state, s
        """
        r = self.get_grid()
        f = self.get_beam(r)
        n_init = np.full_like(r, self.nr)
        # Solve with interval-based sampling
        n_99, depletion_time, nss = self.solve_to_depletion(f, target_fraction=0.01, n_init=n_init, verbose=False)
        self.n = n_99
        return depletion_time

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
        interp_func = CubicSpline(timestamps[:-1], n_center[:-1])

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


def _plot_solution_covergence(iters, norm_array, a=None, b=None, eps=None, fig=None, axes=None, dt=None, times=None):
    if fig is None or axes is None:
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
    else:
        ax, ax_sec = axes
        ax.cla()
        ax_sec.cla()
    ax.semilogy(iters, norm_array, 'o', label='Achieved accuracy')
    if a is not None and b is not None:
        x_fit = np.linspace(0, iters[-1], 300)
        y_fit = np.exp(a) * np.exp(b * x_fit)
        ax.semilogy(x_fit, y_fit, '-', label='Fitted convergence')
        if eps is not None:
            iter_exact = (np.log(eps) - a) / b
            ax.axvline(iter_exact, color='r', linestyle='--', label='Predicted convergence iteration')
            ax.semilogy([iter_exact], [eps], 'rx', label='Predicted convergence point')
    if times is not None:
        try:
            # Add a secondary x-axis (top) showing time in seconds: time = iterations * dt
            # ax_sec = ax.secondary_xaxis('top', functions=(lambda x: x * dt, lambda t: t / dt))
            if axes is None:
                ax_sec = ax.twiny()
            ax_sec.semilogy(times, norm_array, 'x', alpha=0)  # Dummy plot to set scale
            ax_sec.set_xlabel('Time (s)')
            if eps is not None:
                # Use the last few points (e.g., last 5) for better local accuracy near steady state
                n_points = min(len(norm_array), 5)

                # Get arrays for fitting
                y_data = np.log(np.array(norm_array[-n_points:]))
                x_data = np.array(times[-n_points:])

                # Linear regression: y = mx + c
                # Using numpy's polyfit for stability (degree 1)
                slope, intercept = np.polyfit(x_data, y_data, 1)

                # Solve for t: ln(eps) = slope * t + intercept
                # t = (ln(eps) - intercept) / slope
                target_log = np.log(eps)
                time_elapsed_corrected = (target_log - intercept) / slope
                ax_sec.semilogy([time_elapsed_corrected], [eps], '*', alpha=0)
        except Exception:
            # Fallback for older matplotlib: create a twinned top axis and set tick labels manually
            print("Warning: Could not create secondary x-axis.")

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Norm')
    ax.set_title('Solution Convergence')
    ax.legend()
    ax.grid()
    fig.savefig('convergence_plot.png', dpi=300)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    # plt.ion()
    return fig, (ax, ax_sec)


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
