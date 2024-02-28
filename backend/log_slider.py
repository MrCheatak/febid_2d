from math import log
import numpy as np
from matplotlib.widgets import Slider


class LogSlider(Slider):
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, base=10, **kwargs):
        self.base = base
        self._val = 0
        super().__init__(ax, label, np.log10(valmin),
                         np.log10(valmax), valinit=np.log10(valinit),
                         **kwargs)

    def set_val(self, val, log_transform=False):
        val_log = val
        if log_transform:
            val_log = log(val, self.base)
        super().set_val(val_log)
        self.valtext.set_text(self._format(val, log_transform))

    def on_changed(self, func):
        def wrapper(val):
            val = self.base ** val
            func(val)

        super().on_changed(wrapper)

    @property
    def val(self):
        return self.base ** self._val

    @val.setter
    def val(self, x):
        log_transform = False
        if type(x) is tuple:
            x, log_transform = x
        self._val = x
        if log_transform:
            self._val = log(x, self.base)

    def _format(self, val, log_transform=False):
        """Pretty-print *val*."""
        val_log = val
        if not log_transform:
            val_log = self.base ** val
        valfmt = f'{val_log:.3e}'
        return valfmt

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = LogSlider(ax_slider, 'Value', 10, 1e8)


    def update(val):
        # Your update logic here using the logarithmic scale value
        print("Slider Value:", val)


    slider.on_changed(update)

    plt.show()

# Example usage:
