import sqlite3
import numpy as np
from tqdm import tqdm

from processclass import Experiment2D
from program import loop_param

pr = Experiment2D()

pr.n0 = 3.0  # 1/nm^2
pr.F = 2000.0  # 1/nm^2/s
pr.s = 1.0
pr.V = 0.4  # nm^3
pr.tau = 200e-6  # s
pr.D = 2e6  # nm^2/s
pr.sigma = 0.02  #
pr.f0 = 1.0e7
pr.st_dev = 30.0
pr.order = 1
pr.step = 0.1


class SQlite:
    conn = None
    c = None
    db_file = ''
    param_table_name = "params"
    curve_table_name = "curves"
    param_names = None

    def open_file(self, fname, param_names):
        # Define the database file path and table names
        self.db_file = fname
        self.param_names = param_names
        # Connect to the database and create the tables
        self.conn = sqlite3.connect(self.db_file)
        self.c = self.conn.cursor()
        try:
            self.c.execute(
                f"CREATE TABLE {self.param_table_name} ({', '.join([f'{p} REAL' for p in param_names])}, PRIMARY KEY ({', '.join(param_names)}))")
            self.c.execute(
                f"CREATE TABLE {self.curve_table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join([f'{p} REAL' for p in param_names])}, x REAL, y REAL)")
        except sqlite3.OperationalError:
            pass

    def check_exists(self, param_values):
        query = f'SELECT "n0" FROM {self.curve_table_name} WHERE "n0" = ? AND "F" = ? AND "s" = ? AND "tau" = ? AND "D" = ? AND "sigma" = ? AND "f0" = ? AND "st_dev" = ?'
        # self.c.execute(query, param_values)
        # if not self.c.fetchone():
        #     print(f'Data with {param_values} is already present in the database')
        #     return 1
        # else:
        #     return 0

    def save_param_set(self, param_values):
        try:
            self.c.execute(f"INSERT INTO {self.param_table_name} VALUES ({', '.join([str(p) for p in param_values])})")
        except sqlite3.IntegrityError:
            return 1
        return 0
    def save_data(self, x, y, param_values):
        self.c.execute(
            f"INSERT INTO {self.curve_table_name} ({', '.join(self.param_names)}, x, y) VALUES ({', '.join([str(p) for p in param_values])}, ?, ?)",
            (x, y))

    def get_curves_by_parameter(self, parameter_name, **constant_values):
        """
        Retrieve curves from an SQLite3 database by specifying the name of the parameter to inspect and constant values
        for all other parameters.

        Args:
            parameter_name (str): The name of the parameter to inspect.
            constant_values (dict): A dictionary containing the names and values of the parameters to hold constant.

        Returns:
            A list of tuples, where each tuple contains the value of the specified parameter and the corresponding curve.
        """
        # Build a SQL query to retrieve the specified parameter and corresponding curve
        query_params = []
        select_clause = 'SELECT "{}", "x", "y"'.format(parameter_name)
        from_clause = f'FROM {self.curve_table_name}'
        where_clause = 'WHERE ' + ' AND '.join('"{}" = ?'.format(k) for k in constant_values.keys())
        query_params.extend(constant_values.values())
        order_by_clause = 'ORDER BY "{}"'.format(parameter_name)
        query = ' '.join((select_clause, from_clause, where_clause, order_by_clause))

        # Execute the query and extract the parameter values and corresponding curves into a list of tuples
        curves = []
        self.conn = sqlite3.connect(self.db_file)
        self.c = self.conn.cursor()
        self.c.execute(query, query_params)
        data = self.c.fetchall()
        param_vals = np.zeros(len(data))
        for i, row in enumerate(data):
            param_vals[i] = row[0]
            curves.append((np.fromstring(row[1]), np.fromstring(row[2])))

        return param_vals, curves

    def save(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


def generate_db(fname='database.db'):
    param_names = ['n0', 'F', 's', 'tau', 'D', 'sigma', 'f0', 'st_dev', ]
    db = SQlite()
    db.open_file(fname, param_names)
    n0_vals = np.arange(1, 3.5, 0.1).round(1)
    F_vals = np.arange(1000, 4100, 10).round(0)
    s_vals = np.arange(0.1, 1.1, 0.1).round(1)
    tau_vals = np.arange(20e-6, 2e-3, 50e-6).round(6)
    D_vals = np.arange(0, 6e6, 100e4).round(0)
    sigma_vals = np.arange(0.001, 0.1, 0.002).round(3)
    f0_vals = np.float_power(10, np.arange(5, 9.1, 0.1)).round(0)
    st_dev_vals = np.arange(2, 100, 1)
    n0_vals = [3]
    st_dev_vals = [30]
    s_vals = [1]
    F_vals = [1500]
    D_vals = [0]
    # tau_vals = [100e-6]
    f0_vals = [1e7]
    n_passes = 0
    total_iters = len(n0_vals) * len(F_vals) * len(s_vals) * len(tau_vals) * len(D_vals) * len(sigma_vals) * len(
        f0_vals) * len(st_dev_vals)
    iters = 0
    for f0 in f0_vals:
        pr.f0 = f0
        for st_dev in st_dev_vals:
            pr.st_dev = st_dev
            bonds = pr.get_bonds()
            r = np.arange(-bonds, bonds, pr.step)
            f = pr.get_beam(r)
            for D in D_vals:
                pr.D = D
                for tau in tau_vals:
                    pr.tau = tau
                    for s in s_vals:
                        pr.s = s
                        for sigma in sigma_vals:
                            pr.sigma = sigma
                            for F in F_vals:
                                pr.F = F
                                for n0 in n0_vals:
                                    pr.n0 = n0

                                    iters += 1
                                    param_values = [n0, F, s, tau, D, sigma, f0, st_dev]
                                    if not db.save_param_set(param_values):
                                        n_passes += 1
                                        r, R, n = pr.analytic(r, f)
                                        if pr.D > 0:
                                            r, R, n = pr.solve_steady_state(r, f, n_init=n)
                                        if iters%1000 == 0:
                                            print(f'Finished {param_values}    {iters}/{total_iters}')
                                        db.save_data(r, n, param_values)
                                    else:
                                        if iters%10000 == 0:
                                            print(f'Skipping {param_values}    {iters}/{total_iters}')
                                    if n_passes > 2000:
                                        db.save()
                                        n_passes = 0
    db.save()
    db.close()


if __name__ == '__main__':
    generate_db()
