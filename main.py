from processclass import Experiment2D
from timeit import default_timer as dt

pr = Experiment2D()

pr.n0 = 2.7  # 1/nm^2
pr.F = 730.0  # 1/nm^2/s
pr.s = 0.0062
pr.V = 0.05  # nm^3
pr.tau = 1000e-6  # s
pr.D = 1e6  # nm^2/s
pr.sigma = 0.02  # nm^2
pr.f0 = 2e6
pr.fwhm = 50  # nm
pr.order = 1
pr.step = 0.5  # nm
pr.beam_type = 'gauss'
pr.order = 4



if __name__ == '__main__':
    pr.backend = 'gpu'
    pr.solve_steady_state(eps=1e-8, progress=True)
    pr.plot('R')
    pr.backend = 'gpu'



