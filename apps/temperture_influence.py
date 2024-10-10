"""
The app is a simple tool to visualize the temperature influence on the depletion and diffusive replenishment
under identical conditions. The influence is modeled through the arrhenius equations for diffusion and
residence time. The user can adjust the activation energies and pre-exponential factors for both processes to assess
the impact on the position and arrangement of the points and subsequent change of the power equation that is used
to fit the points.
It is also possible to add custom points to the plot by entering the coordinates in the text box. This is useful for
comparing the model with experimental data.
Exponential and power functions can be selected for fitting the points. Power function fits well for most cases,
exponential function is best for when the points are relatively close to origin.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons
from scipy.optimize import curve_fit
from backend.processclass import Experiment1D
from backend.log_slider import LogSlider

kb = 8.617333262145e-5


def tau_T(k0, Ea, T):
    return 1 / k0 * np.exp(Ea / (kb * T))


def D_T(D0, Ed, T):
    return D0 * np.exp(-Ed / (kb * T))


def power_func(x, a, b):
    return a * np.power(x, b)


def power_func_label(a, b):
    return f'$y = {a:.4f}x^{{{b:.4f}}}$'


def exponential_func(x, a, b):
    return a * np.exp(b * x)


def exponential_func_label(a, b):
    return f'$y = {a:.4f}e^{{{b:.4f}x}}$'


def fit_points(x, y, func):
    popt, _ = curve_fit(func, x, y)
    a, b = popt
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = func(x_fit, a, b)
    return x_fit, y_fit, a, b


pr = Experiment1D()
pr.n0 = 2.8  # 1/nm^2
pr.F = 230  # 1/nm^2/s
pr.s = 0
pr.V = 0.01  # nm^3
pr.tau = 500e-6  # s
pr.D = 5e5  # nm^2/s
pr.sigma = 0.012  # nm^2
pr.fwhm = 100  # nm
pr.f0 = 1e7
pr.step = pr.fwhm // 200  # nm
pr.beam_type = 'gauss'
pr.order = 1

Ts = np.linspace(5, 50, 10) + 273.15

k0 = 1e15
Ea = 0.7
taus = tau_T(k0, Ea, Ts)

D0 = 46e6
Ed = 0.09
Ds = D_T(D0, Ed, Ts)

pr.tau = taus
pr.D = Ds

fig, ax = plt.subplots(figsize=(12, 11))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.4)

sc = ax.scatter(pr.p_o, pr.tau_r)

annotations = [ax.annotate(f'{T - 273.15:.2f}', (pr.p_o[i], pr.tau_r[i]), annotation_clip=False) for i, T in
               enumerate(Ts)]

axcolor = 'lightgoldenrodyellow'
slider_offset = 0.01
ax_Ea = plt.axes([0.1, slider_offset, 0.65, 0.03], facecolor=axcolor)
ax_k0 = plt.axes([0.1, slider_offset + 0.05, 0.65, 0.03], facecolor=axcolor)
ax_Ed = plt.axes([0.1, slider_offset + 0.1, 0.65, 0.03], facecolor=axcolor)
ax_D0 = plt.axes([0.1, slider_offset + 0.15, 0.65, 0.03], facecolor=axcolor)
ax_sigma = plt.axes([0.1, slider_offset + 0.2, 0.65, 0.03], facecolor=axcolor)
ax_textbox = plt.axes([0.1, slider_offset + 0.25, 0.65, 0.03], facecolor=axcolor)
ax_radio = plt.axes([0.865, 0.1, 0.14, 0.14], facecolor=axcolor)

s_Ea = Slider(ax_Ea, '$E_a, eV$', 0.1, 1.0, valinit=Ea)
s_k0 = LogSlider(ax_k0, '$k_0, Hz$', 1e12, 1e18, valinit=k0)
s_Ed = Slider(ax_Ed, '$E_D, eV$', 0.01, 0.2, valinit=Ed)
s_D0 = LogSlider(ax_D0, '$D_0, nm^{-2}$', 1e4, 1e9, valinit=D0)
s_sigma = LogSlider(ax_sigma, '$\\sigma, nm^2$', 0.00001, 1, valinit=pr.sigma)
text_box = TextBox(ax_textbox, 'Add Point (x,y)', initial="")
radio = RadioButtons(ax_radio, ('Power', 'Exponential'))

D_display = ax.text(0.01, 0.95, f'D(25) = {pr.D[4]:.1e} [$nm^{-2}$]', transform=ax.transAxes)
tau_display = ax.text(0.01, 0.9, f'$\\tau$(25) = {pr.tau[4] * 1e3:.3f} [ms]', transform=ax.transAxes)

fitting_func = power_func
label_func = power_func_label

x_fit, y_fit, a, b = fit_points(pr.p_o, pr.tau_r, fitting_func)
[line] = ax.plot(x_fit, y_fit, color='red', label=label_func(a, b))
ax.legend()


def tau_text():
    return f'$\\tau$(25) = {pr.tau[4] * 1e3:.3f} [ms]'


def D_text():
    return f'D(25) = {pr.D[4]:.1e} [$nm^{-2}$]'


def update(val):
    Ea = s_Ea.val
    k0 = s_k0.val
    Ed = s_Ed.val
    D0 = s_D0.val
    sigma = s_sigma.val
    taus = tau_T(k0, Ea, Ts)
    Ds = D_T(D0, Ed, Ts)
    pr.tau = taus
    pr.D = Ds
    pr.sigma = sigma
    sc.set_offsets(np.c_[pr.p_o, pr.tau_r])
    for i, T in enumerate(Ts):
        annotations[i].set_position((pr.p_o[i], pr.tau_r[i]))
    x_fit, y_fit, a, b = fit_points(pr.p_o, pr.tau_r, fitting_func)
    line.set_data(x_fit, y_fit)
    line.set_label(label_func(a, b))
    ax.legend()
    D_display.set_text(D_text())
    tau_display.set_text(tau_text())
    fig.canvas.draw_idle()


def add_point(text):
    try:
        x, y = map(float, text.split(','))
        ax.scatter(x, y, color='red')
        fig.canvas.draw_idle()
    except ValueError:
        print("Invalid input. Please enter coordinates in the format 'x,y'.")


def select_func(label):
    global fitting_func, label_func
    if label == 'Power':
        fitting_func = power_func
        label_func = power_func_label
    else:
        fitting_func = exponential_func
        label_func = exponential_func_label
    update(0)


s_Ea.on_changed(update)
s_k0.on_changed(update)
s_Ed.on_changed(update)
s_D0.on_changed(update)
s_sigma.on_changed(update)
text_box.on_submit(add_point)
radio.on_clicked(select_func)

update(0)

ax.set_xlim(0, 3)
ax.set_ylim(1, 5000)

ax.set_ylabel('Depletion $\\tilde{\\tau}_r$')
ax.set_xlabel('Diffusive replenishment $\\tilde{p}_{out}$')
ax.set_title('Temperature dependence (⁰C) of depletion and diffusive replenishment')

plt.show()
