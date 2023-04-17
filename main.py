from program import analytic, numeric, numeric_gpu, get_all, plot_line
import program
import numpy as np

program.n0 = 3.0  # 1/nm^2
program.F = 2000.0  # 1/nm^2/s
program.s = 1.0
program.V = 0.4  # nm^3
program.tau = 200e-6  # s
program.D = 2e5  # nm^2/s
program.sigma = 0.02  #
program.f0 = 1.0e7
program.st_dev = 30.0
program.order = 1

program.step = 0.1

program.fwhm = 30.0
program.st_dev = program.fwhm/2.355

get_all()

if __name__ == '__main__':
    r = np.arange(-program.st_dev * 5, program.st_dev * 5, program.step)
    R_a, n_a = analytic(r)
    D_vals = [0.0, 5e4, 1e5, 2e5, 3e5, 5e5, 1e6, 5e6]
    R = []
    p = []
    tau = []
    phi1 = []
    phi2 = []
    for d in D_vals:
        program.D = d
        get_all()
        p.append(program.p_out)
        tau.append(program.tau_r)
        phi1.append(program.phi_1)
        phi2.append(program.phi_2)
        R_1, n = numeric(r, n_a)
        R.append(R_1)

    # r = r*2 /program.fwhm
    plot_line(r, R_a, 'Analytic', marker='.')
    for i, data in enumerate(R):
        label = f'ρ={p[i]:.3f} ' \
                f'τ={tau[i]:.3f}\n' \
                f'φ1={phi1[i]:.3f} ' \
                f'φ2={phi2[i]:.3f}\n'
        plot_line(r, data, f'Numeric, D={D_vals[i]:.0e}\n'+label)
    program.ax.legend(handlelength=4)
    program.plt.legend(fontsize=8)
    program.plt.show()
