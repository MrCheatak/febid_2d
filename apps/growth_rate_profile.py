from math import log

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from backend.processclass import Experiment1D
from backend.electron_flux import EFluxEstimator
from backend.log_slider import LogSlider

prs = []
fwhms = np.array([300, 800, 1000, 1200, 1400, 1600])
ies = np.array([98e-12, 400e-12, 1600e-12, 6300e-12])
es = EFluxEstimator()
es.ie = ies
es.fwhm = fwhms
es.yld = 0.68
ess = []

pr = Experiment1D()

pr.name = 'Co3Fe'
pr.n0 = 2.8  # 1/nm^2
pr.F = 730  # 1/nm^2/s
pr.s = 0.0062
pr.V = 0.05  # nm^3
pr.tau = 4e-6  # s
pr.D = 0  # nm^2/s
pr.sigma = 0.002  # nm^2
pr.f0 = 1e7
pr.fwhm = 300  # nm
pr.order = 1
pr.beam_type = 'super_gauss'
pr.step = 1  # nm
pr.backend = 'cpu'

n = es.ie.size  # total number of points

fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.55, top=0.99)  # Adjust the position of the sliders

### Setting up sliders
initial_param1 = 1
initial_param2 = 1700
initial_param3 = 2.8
initial_param4 = 200e-6
initial_param5 = 0.022
initial_param6 = 5e6
initial_param7 = 100
initial_param8 = 1
initial_param9 = 1

ax_param1 = plt.axes([0.2, 0.4, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_param2 = plt.axes([0.2, 0.35, 0.63, 0.03], facecolor='lightgoldenrodyellow')
ax_param3 = plt.axes([0.2, 0.3, 0.58, 0.03], facecolor='lightgoldenrodyellow')
ax_param4 = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_param5 = plt.axes([0.2, 0.2, 0.63, 0.03], facecolor='lightgoldenrodyellow')
ax_param6 = plt.axes([0.2, 0.15, 0.58, 0.03], facecolor='lightgoldenrodyellow')
ax_param7 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_param8 = plt.axes([0.2, 0.05, 0.63, 0.03], facecolor='lightgoldenrodyellow')
ax_param9 = plt.axes([0.2, 0.0, 0.58, 0.03], facecolor='lightgoldenrodyellow')

s_param1 = LogSlider(ax_param1, r'Stick. coeff. (s)', 1e-5, 1, valinit=initial_param1)
s_param2 = LogSlider(ax_param2, r'Gas flux (F), $nm^{-2}\cdot s^{-1}$', 1, 10000, valinit=initial_param2)
s_param3 = Slider(ax_param3, r'Max. coverage ($n_{0}$), $nm^{-2}$', 0.1, 10, valinit=initial_param3)
s_param4 = LogSlider(ax_param4, r'Res. time ($\tau_{r}$), s', 20e-6, 0.1, valinit=initial_param4)
s_param5 = Slider(ax_param5, r'Diss. c.-sect. ($\sigma$), $nm^2$', 1e-4, 5e-1, valinit=initial_param5)
s_param6 = LogSlider(ax_param6, r'Electron flux ($f_{0}$), $nm^{-2}\cdot s^{-1}$', 1e2, 1e9, valinit=initial_param6)
s_param7 = Slider(ax_param7, r'Beam FWHM, nm', 5, 1500, valinit=initial_param7)
s_param8 = Slider(ax_param8, r'Beam flatness m', 1, 10, valinit=initial_param8)
s_param9 = LogSlider(ax_param9, r'Diff. coeff. (D), $1/{nm^2 * s}$', 1, 1e7, valinit=initial_param9)

sliders = [s_param1, s_param2, s_param3, s_param4, s_param5, s_param6, s_param7, s_param8, s_param9]
params = ['s', 'F', 'n0', 'tau', 'sigma', 'f0', 'fwhm', 'order', 'D']

r_max_display = ax.text(0.01, 0.93, '$r_{max}$: 0', fontsize=8, transform=ax.transAxes)
R_ind_display = ax.text(0.01, 0.87, '$R_{ind}$: 0', fontsize=8, transform=ax.transAxes)
fwhm_d_display = ax.text(0.01, 0.81, '$FWHM_D$: 0', fontsize=8, transform=ax.transAxes)
phi_display = ax.text(0.01, 0.75, '$\phi$: 0', fontsize=8, transform=ax.transAxes)
tau_r_display = ax.text(0.01, 0.67, '$\\tilde{\\tau}$: 0', fontsize=8, transform=ax.transAxes)
p_o_display = ax.text(0.01, 0.60, '$\\tilde{p}_{out}$: 0', fontsize=8, transform=ax.transAxes)


def r_max_text():
    return f'$r_{{max}}$: {pr.r_max_n:.3f}'


def R_ind_text():
    return f'$R_{{ind}}$: {pr.R_ind:.3f}'

def fwhm_d_text():
    return f'$FWHM_D$: {pr.fwhm_d:.1f} nm'

def phi_text():
    return f'$\phi$: {pr.fwhm_d/pr.fwhm:.3f}'

def tau_r_text():
    return f'$\\tilde{{\\tau}}$: {pr.tau_r:.1f}'

def p_o_text():
    return f'$\\tilde{{p}}_{{out}}$: {pr.p_o:.2f}'


def events_switch(flag):
    for slider in sliders:
        slider.eventson = flag


def update(val):
    for i in range(len(sliders)):
        setattr(pr, params[i], sliders[i].val)
    pr.step = pr.fwhm / 200
    # if pr.D > 1e6:
    #     pr.backend = 'gpu'
    # if pr.D <= 1e6:
    #     pr.backend = 'cpu'
    pr.solve_steady_state()
    x, y = pr.r, pr.R
    r_max_display.set_text(r_max_text())
    R_ind_display.set_text(R_ind_text())
    fwhm_d_display.set_text(fwhm_d_text())
    phi_display.set_text(phi_text())
    tau_r_display.set_text(tau_r_text())
    p_o_display.set_text(p_o_text())
    line.set_ydata(y)
    line.set_xdata(x)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()


for slider in sliders:
    slider.on_changed(update)

### Setting up Button for PtC5
button1_ax = plt.axes((0.9, 0.9, 0.1, 0.05))
button1 = Button(button1_ax, 'PtC5', color='orange', hovercolor='white')


def set_ptc5(event):
    pr1 = pr.__copy__()
    pr1.n0 = 2.8  # 1/nm^2
    pr1.F = 1700.0  # 1/nm^2/s
    pr1.s = 1
    pr1.tau = 400e-6  # s
    pr1.D = 3.5e5  # nm^2/s
    pr1.sigma = 0.022  # nm^2
    events_switch(False)
    sliders[2].set_val(pr1.n0)
    sliders[1].set_val(pr1.F, True)
    sliders[0].set_val(pr1.s, True)
    sliders[3].set_val(pr1.tau, True)
    sliders[4].set_val(pr1.sigma)
    sliders[8].set_val(pr1.D, True)
    update(0)
    events_switch(True)


button1.on_clicked(set_ptc5)

### Setting up button for W(CO)6
button2_ax = plt.axes((0.9, 0.8, 0.1, 0.05))
button2 = Button(button2_ax, 'W(CO)6', color='green', hovercolor='white')


def set_wco6(event):
    pr1 = pr.__copy__()
    pr1.n0 = 2.5  # 1/nm^2
    pr1.F = 1700.0  # 1/nm^2/s
    pr1.s = 0.025
    pr1.tau = 3200e-6  # s
    pr1.D = 6.4e6  # nm^2/s
    pr1.sigma = 0.5  # nm^2
    events_switch(False)
    sliders[2].set_val(pr1.n0)
    sliders[1].set_val(pr1.F)
    sliders[0].set_val(pr1.s)
    sliders[3].set_val(pr1.tau, True)
    sliders[4].set_val(pr1.sigma)
    sliders[8].set_val(pr1.D, True)
    update(0)
    events_switch(True)


button2.on_clicked(set_wco6)

# Initial curve
pr.solve_steady_state()
x = pr.r
y = pr.R
line, = ax.plot(x, y)
update(0)

ax.set_xlabel('r')
ax.set_ylabel('R/sFV')
plt.show()
a = 0
