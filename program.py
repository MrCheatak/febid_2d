import numpy as np
import numexpr_mod as ne
from math import sqrt, log
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyopencl as cl
from pycl_test import cl_boilerplate


def get_fwhm():
    global fwhm
    fwhm = 2.355 * st_dev


def get_st_dev():
    global st_dev
    st_dev = fwhm/2.355


def get_dt_diff():
    global dt_dif
    if D > 0:
        dt_dif = step ** 2 / (2 * D)
    else:
        dt_dif = 1
    get_dt()
    return dt_dif


def get_dt_diss():
    global dt_diss
    dt_diss = 1 / sigma / f0
    get_dt()
    return dt_diss


def get_dt_des():
    global dt_des
    dt_des = tau
    get_dt()
    return dt_des


def get_dt():
    global dt
    dt = np.min([dt_des, dt_diss, dt_dif]) * 0.7
    return dt


def get_kd():
    global kd
    kd = s * F / n0 + 1 / tau + sigma * f0
    get_nd()
    get_tau_in()
    return kd


def get_kr():
    global kr
    kr = s * F / n0 + 1 / tau
    get_nr()
    get_tau_out()
    return kr


def get_nd():
    global nd
    nd = s * F / kd
    return nd


def get_nr():
    global nr
    nr = s * F / kr
    return nr


def get_tau_in():
    global tau_in
    tau_in = 1 / kd
    get_tau_r()
    get_p_in()
    return tau_in


def get_tau_out():
    global tau_out
    tau_out = 1 / kr
    get_tau_r()
    return tau_out


def get_tau_r():
    global tau_r
    tau_r = tau_out / tau_in
    get_phi_1()
    return tau_r


def get_p_in():
    global p_in
    p_in = sqrt(D * tau_in)
    return p_in


def get_p_out():
    global p_out
    p_out = 2 * p_in / fwhm
    get_phi_2()
    return p_out


def get_phi_1():
    global phi_1
    phi_1 = sqrt(log(1 + tau_r, 2))
    return phi_1


def get_phi_2():
    global phi_2
    try:
        phi_2 = sqrt(log(2 + p_out ** -2))
    except ZeroDivisionError:
        phi_2 = np.nan
    return phi_2


def get_all():
    get_fwhm()
    get_dt_diff()
    get_dt_des()
    get_dt_diss()
    get_kd()
    get_kr()
    get_p_out()


n0 = 3.0  # 1/nm^2
F = 2000.0  # 1/nm^2/s
s = 1.0
V = 0.4  # nm^3
tau = 200e-6  # s
D = 2e5  # nm^2/s
sigma = 0.02  #
f0 = 1.0e7
st_dev = 20.0
order = 1.0

fwhm = 2.355 * st_dev

kd = s * F / n0 + 1 / tau + sigma * f0
kr = s * F / n0 + 1 / tau

nd = s * F / kd
nr = s * F / kr

tau_in = 1 / kd
tau_out = 1 / kr
tau_r = tau_out / tau_in

p_in = sqrt(D * tau_in)
p_out = 2 * p_in / fwhm

phi_1 = sqrt(log(1 + tau_r, 2))
phi_2 = sqrt(log(2 + p_out ** -2))

start = -st_dev * 5
end = st_dev * 5
step = 0.3

dt_des = tau
dt_diss = 1 / sigma / f0
dt_dif = 0
dt = 0
get_dt_diff()
get_dt()

n = 0.0
f = 1e5
n_D = 1e-5
expr = ne.cache_expression("(s*F*(1-n/n0) - n/tau - sigma*f*n + n_D*D/step**2)*dt + n", 'r_e')

fig, ax = plt.subplots()


def plot_line(x, y, name=None, marker=None):
    line, = ax.plot(x, y, label=name, marker=marker)


def fit_exponential(x0, y0):
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


def get_gauss(r):
    r = r.astype(dtype=np.longdouble)
    return f0 * np.exp((-r ** 2 / (2 * st_dev ** 2)) ** order)


def diffusion(n):
    n_out = np.copy(n)
    n_out[0] = 0
    n_out[-1] = 0
    n_out[1:-1] *= -2
    n_out[1:-1] += n[2:]
    n_out[1:-1] += n[:-2]
    return n_out


def numeric(r, n_init, eps=1e-8):
    n_iters = int(1e9)
    n = np.copy(n_init)
    n_check = np.copy(n_init)
    f = get_gauss(r)

    base_step = 100
    skip_step = base_step * 5
    skip = skip_step  # next planned accuracy check iteration
    prediction_step = skip_step * 3
    n_predictions = 0
    norm = 1  # achieved accuracy
    norm_array = []
    iters = []
    for i in tqdm(range(n_iters)):
        if i % skip == 0 and i != 0:  # skipping array copy
            n_check[...] = n

        n_D = diffusion(n)
        ne.re_evaluate('r_e', out=n)
        if n.max() > n0 or n.min() < 0:
            raise ValueError('Solution unstable!')

        if i % skip == 0 and i != 0:  # skipping achieved accuracy evaluation

            norm = (np.linalg.norm(n[1:] - n_check[1:]) / np.linalg.norm(n[1:]))
            norm_array.append(norm)  # recording achieved accuracy for fitting
            iters.append(i)
            skip += skip_step
        if eps > norm:
            print(f'Reached solution with an error of {norm:.3e}')
            break
        if i % prediction_step == 0 and i != 0:
            a, b = fit_exponential(iters, norm_array)
            skip = int((np.log(eps) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
            prediction_step = skip  # next prediction will be after another norm is calculated
            n_predictions += 1
    R = n * sigma * f / s / F
    return R, n


def numeric_gpu(r, n_init, eps=1e-12):

    s_ = np.array([s])
    F_ = np.array([F])
    n0_ = np.array([n0])
    tau_ = np.array([tau])
    sigma_ = np.array([sigma])
    D_ = np.array([D])
    step_ = np.array([step])
    dt_ = np.array([dt])

    n_iters = int(1e9)
    n = np.copy(n_init)
    n_check = np.copy(n_init)
    f = get_gauss(r)
    n_D = np.zeros_like(n)

    base_step = 100
    skip_step = base_step * 5
    skip = skip_step  # next planned accuracy check iteration
    prediction_step = skip_step * 3
    n_predictions = 0
    norm = 1  # achieved accuracy
    norm_array = []
    iters = []

    context, prog, queue = cl_boilerplate()
    index = n.nonzero()[0]
    index = index[1:-1].astype(int)
    n_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=n.nbytes)
    n_D_dev = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=n.nbytes)
    f_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=f.nbytes)
    index_dev = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=index.nbytes)

    cl.enqueue_copy(queue, n_dev, n)
    cl.enqueue_copy(queue, index_dev, index)
    cl.enqueue_copy(queue, f_dev, f)

    for i in tqdm(range(n_iters)):
        if i % skip == 0 and i != 0:  # skipping array copy
            cl.enqueue_copy(queue, n_check, n_dev)

        prog.stencil(queue, index.shape, None, n_dev, index_dev, n_D_dev)
        prog.reaction_equation(queue, n.shape, None, n_dev, s_, F_, n0_, tau_, sigma_, f_dev, D_, n_D_dev, step_, dt_)
        # cl.enqueue_copy(queue, n, n_dev)

        if n.max() > n0 or n.min() < 0:
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
            a, b = fit_exponential(iters, norm_array)
            skip = int((np.log(eps) - a) / b) + skip_step * n_predictions  # making a prediction with overcompensation
            prediction_step = skip  # next prediction will be after another norm is calculated
            n_predictions += 1

    cl.enqueue_copy(queue, n, n_dev)
    R = n * sigma * f / s / F
    return R, n

def analytic(r):
    r_norm = 2 * r / fwhm
    f = get_gauss(r)
    t_eff = (s * F / n0 + 1 / tau + sigma * f) ** -1
    n_a = s * F * t_eff
    R_a = t_eff * sigma * f
    return R_a, n_a


if __name__ == '__main__':
    r = np.arange(-st_dev * 5, st_dev * 5, step)
    R_a, n_a = analytic(r)
    D_vals = [0, 5e4, 1e5, 2e5, 3e5, 5e5, 1e6]
    R = []
    for d in D_vals:
        D = d
        get_all()
        R_1, n = numeric(r, n_a)
        R.append(R_1)

    plot_line(r, R_a, 'Analytic', marker='.')
    for data, d in zip(R, D_vals):
        plot_line(r, data, f'Numeric, D={d:.0e}')
    ax.legend(handlelength=4)
    plt.show()
