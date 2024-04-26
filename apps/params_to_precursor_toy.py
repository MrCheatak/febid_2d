import math
import matplotlib.image as mpimg
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QGridLayout, QWidget, QLineEdit, QLabel, \
    QPushButton, QHBoxLayout, QFileDialog, QTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.interpolate import LinearNDInterpolator
from backend import adaptive_tools

from backend.model import Model, keys_to_string


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=200):
        self.distance_threshold = 0.1
        self.parent = parent
        self._dragging_point = None
        self._points = {}
        self._lines = []
        self.extent = []
        self.fig_kwargs = {'layout': 'constrained', 'figsize': (width, height), 'dpi': dpi}
        self._fig, self._axes = plt.subplots(1, 2, **self.fig_kwargs)
        self._fig: Figure
        super(MplCanvas, self).__init__(self._fig)
        self.initial_positions = np.array([(1.37, 1100, 283), (0.25, 290, 298), (0.18, 38, 313)])

        self.mpl_connect('button_press_event', self._on_click)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)

    def plot_maps_from_files(self, file_name1, file_name2, extent):
        """
        Plots maps from image files onto the axes.

        :param file_name1: The file path of the first image.
        :param file_name2: The file path of the second image.
        :param extent: The extent of the images to be plotted.
        """
        self._fig:Figure
        self._fig.clear()
        self._axes = self._fig.subplots(1, 2)

        self.extent = extent
        # img = crop_image(file_name2)
        # img1 = mpimg.imread(file_name1)
        # img2 = mpimg.imread(file_name2)
        img1, extent1 = self.load_from_interpolator(file_name1)
        img2, extent2 = self.load_from_interpolator(file_name2)
        imshow1 = self._axes[0].imshow(img1, extent=extent1, aspect='auto', cmap='magma')
        imshow2 = self._axes[1].imshow(img2, extent=extent2, aspect='auto', cmap='magma')
        self._fig.colorbar(imshow1, ax=self._axes[0])
        self._fig.colorbar(imshow2, ax=self._axes[1])
        x_init = self.initial_positions.T[0]
        y_init = self.initial_positions.T[1]
        _line1, = self._axes[0].plot(x_init, y_init, 'green', marker='o', markersize=2,
                                     linewidth=0.5)  # initial positions of the points
        _line2, = self._axes[1].plot(x_init, y_init, 'green', marker='o', markersize=2,
                                     linewidth=0.5)  # initial positions of the points
        self._lines = [_line1, _line2]
        self.draw_idle()
        self.update_figure()

    def compute_initial_figure(self):
        img1 = mpimg.imread(
            r'/pics/R_ind_big_test.png')  # replace with your image file
        img2 = mpimg.imread(r'/pics/r_max_big_test.png')
        self._axes[0].imshow(img1, extent=[0, 1, 0, 1])
        self._axes[1].imshow(img2, extent=[0, 1, 0, 1])
        x_init = self.initial_positions.T[0]
        y_init = self.initial_positions.T[1]
        _line1, = self._axes[0].plot(x_init, y_init, 'white', marker='o', markersize=2,
                                     linewidth=0.5)  # initial positions of the points
        _line2, = self._axes[1].plot(x_init, y_init, 'white', marker='o', markersize=2,
                                     linewidth=0.5)  # initial positions of the points
        self._lines = [_line1, _line2]
        return self._lines

    def update_figure(self):
        if not self._points:
            for line in self._lines:
                line.set_data([], [])
        else:
            i, xy = zip(*self._points.items())
            x, y = zip(*xy)
            for line in self._lines:
                line.set_data(x, y)
        self._axes[0].draw_artist(self._lines[0])
        self._axes[1].draw_artist(self._lines[1])
        # self.update()
        # self.flush_events()
        self.blit(self._fig.bbox)

    def _find_neighbor_point(self, event):
        u""" Find point around mouse position

        :rtype: ((int, int)|None)
        :return: (x, y) if there are any point around mouse else None
        """
        nearest_point = None
        min_distance = math.sqrt(2 * (100 ** 2))
        for i, xy in self._points.items():
            x, y = xy
            xmin, xmax, ymin, ymax = self.extent
            distance = math.hypot((event.xdata - x) / (xmax - xmin), (event.ydata - y) / (ymax - ymin))
            if distance < min_distance:
                min_distance = distance
                nearest_point = (i, (x, y))
        if min_distance < self.distance_threshold:
            return nearest_point
        return None

    def _on_click(self, event):
        """ callback method for mouse click event

        :type event: MouseEvent
        """
        # left click
        if event.button == 1 and event.inaxes in [*self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            # else:
            #     self._add_point(event)
            self.update_figure()

    def _on_release(self, event):
        """ callback method for mouse release event

        :type event: MouseEvent
        """
        if event.button == 1 and event.inaxes in [*self._axes] and self._dragging_point:
            self._dragging_point = None
            self.update_figure()
            self.draw()

    def _on_motion(self, event):
        """ callback method for mouse motion event

        :type event: MouseEvent
        """
        if not self._dragging_point:
            return
        if event.xdata is None or event.ydata is None:
            return
        i = self._dragging_point[0]
        self._update_point(event.xdata, event.ydata, i)
        self.parent.update_fields(self._points, i)
        self.parent.recalculate_precursor()
        self.update_figure()

    def _add_point(self, x, y=None, i=None):
        if isinstance(x, MouseEvent):
            x, y = float(x.xdata), float(x.ydata)
        self._points[i] = (x, y)
        return i, (x, y)

    def _remove_point(self, x, _):
        if x in self._points:
            self._points.pop(x)

    def _update_point(self, x, y, i):
        self._points[i] = (x, y)

    def load_from_interpolator(self, fname):
        learner = adaptive_tools.learner_load_full(fname)
        data = learner.to_numpy()
        interp = LinearNDInterpolator(data[:, 0:2], data[:, 2], rescale=True)
        x, y = adaptive_tools.generate_lin_grid(*learner.bounds, 3000)
        z = interp(x, y)[::-1]
        extent = [learner.bounds[0][0], learner.bounds[0][1], learner.bounds[1][0], learner.bounds[1][1]]
        return z, extent


class ApplicationWindow(QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.setWindowTitle("My Application")

        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)
        hlayout1 = QHBoxLayout(self.main_widget)
        button1 = QPushButton("Open peak pos. map", self)
        button1.clicked.connect(self.open_map)
        self.file_view1 = QLineEdit(self)
        self.file_view1.setReadOnly(True)
        button2 = QPushButton("Open indent depth map", self)
        button2.clicked.connect(self.open_map)
        self.file_view2 = QLineEdit(self)
        self.file_view2.setReadOnly(True)
        hlayout1.addWidget(button1)
        hlayout1.addWidget(self.file_view1)
        hlayout1.addWidget(button2)
        hlayout1.addWidget(self.file_view2)
        self.layout.addLayout(hlayout1)
        hlayout2 = QHBoxLayout(self.main_widget)
        x_label = QLabel("X axis bounds", self)
        self.x_min = QLineEdit(self)
        self.x_min.setText('0')
        self.x_max = QLineEdit(self)
        self.x_max.setText('4')
        y_label = QLabel("Y axis bounds", self)
        self.y_min = QLineEdit(self)
        self.y_min.setText('1')
        self.y_max = QLineEdit(self)
        self.y_max.setText('300')
        button_plot = QPushButton("Plot", self)
        button_plot.clicked.connect(self.plot_maps)
        hlayout2.addWidget(x_label)
        hlayout2.addWidget(self.x_min)
        hlayout2.addWidget(self.x_max)
        hlayout2.addWidget(y_label)
        hlayout2.addWidget(self.y_min)
        hlayout2.addWidget(self.y_max)
        hlayout2.addWidget(button_plot)
        self.layout.addLayout(hlayout2)
        hlayout3 = QHBoxLayout(self.main_widget)


        self.x_min.setText('0')
        self.x_max.setText('4')
        self.y_min.setText('1')
        self.y_max.setText('1500')

        instructions = QLabel("Click on the point and drag it to the new position. "
                              "Use Pan + L. click to move around the plot, use Pan + R. click and move mouse diagonally"
                              " to Zoom in/out", self)
        self.layout.addWidget(instructions)

        self.canvas = MplCanvas(self, width=10, height=15, dpi=200)
        self.canvas.parent = self
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(NavigationToolbar(self.canvas, self))

        self.input_fields = []
        grid_layout = QGridLayout()
        label_depletion = QLabel("Depletion", self)
        label_diff_replenishment = QLabel("Diff. replenishment", self)
        label_temp = QLabel("Temperature", self)
        grid_layout.addWidget(label_depletion, 0, 1)
        grid_layout.addWidget(label_diff_replenishment, 0, 2)
        grid_layout.addWidget(label_temp, 0, 3)
        filed_labels = ['rho', 'tau', 'T']
        for i in range(3):
            row = []
            label = QLabel(f"Point {i + 1}", self)
            grid_layout.addWidget(label, i + 1, 0)
            self.canvas._points[i] = self.canvas.initial_positions[i, :2]
            for j in range(3):
                line_edit = QLineEdit(self)
                field_name = filed_labels[j] + str(i + 1)
                line_edit.setObjectName(field_name)
                line_edit.setText(str(self.canvas.initial_positions[i, j]))
                line_edit.textEdited.connect(self.update_plot)
                line_edit.editingFinished.connect(self.recalculate_precursor)
                grid_layout.addWidget(line_edit, i + 1, j + 1)
                row.append(line_edit)
            self.input_fields.append(row)
        self.layout.addLayout(grid_layout)

        Ea_label = QLabel("Ea", self)
        Ea_view = QLineEdit(self)
        k0_label = QLabel("k0", self)
        k0_view = QLineEdit(self)
        sigma_label = QLabel("sigma", self)
        sigma_view = QLineEdit(self)
        ED_label = QLabel("ED", self)
        ED_view = QLineEdit(self)
        D0_label = QLabel("D0", self)
        D0_view = QLineEdit(self)
        Ea_view.setObjectName('Ea')
        k0_view.setObjectName('k0')
        sigma_view.setObjectName('sigma')
        ED_view.setObjectName('ED')
        D0_view.setObjectName('D0')
        Ea_view.setReadOnly(True)
        k0_view.setReadOnly(True)
        sigma_view.setReadOnly(True)
        ED_view.setReadOnly(True)
        D0_view.setReadOnly(True)
        hlayout3.addWidget(Ea_label)
        hlayout3.addWidget(Ea_view)
        hlayout3.addWidget(k0_label)
        hlayout3.addWidget(k0_view)
        hlayout3.addWidget(sigma_label)
        hlayout3.addWidget(sigma_view)
        hlayout3.addWidget(ED_label)
        hlayout3.addWidget(ED_view)
        hlayout3.addWidget(D0_label)
        hlayout3.addWidget(D0_view)
        self.layout.addLayout(hlayout3)
        self.result_fields = [Ea_view, k0_view, sigma_view, ED_view, D0_view]

        self.comments_view = QTextEdit(self)
        self.comments_view.setReadOnly(True)
        self.comments_view.setMinimumSize(200, 90)
        self.layout.addWidget(self.comments_view)

        self.setCentralWidget(self.main_widget)

        self.model = Model()
        self.model_params = {'J': 5400, 's': 0.0004, 'n0': 2.7, 'f0': 9e5, 'FWHM': 1400}
        self.model.set_params(**self.model_params)

    def update_plot(self):
        xdata = [float(field[0].text()) for field in self.input_fields]
        ydata = [float(field[1].text()) for field in self.input_fields]
        self.canvas._points = {i: (x, y) for i, (x, y) in enumerate(zip(xdata, ydata))}
        self.canvas.update_figure()

    def update_fields(self, points, index=None):
        if index is not None:
            self.input_fields[index][0].setText(f'{points[index][0]:.4f}')
            self.input_fields[index][1].setText(f'{points[index][1]:.4f}')
            return
        for i, (x, y) in points.items():
            self.input_fields[i][0].setText(f'{x:.4f}')
            self.input_fields[i][1].setText(f'{y:.4f}')

    def open_map(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open map image',
                                                   filter='Interpolator files (*.int)')
        button = self.sender()
        if button.text() == 'Open peak pos. map':
            index = 0
            self.file_view1.setText(file_name)
        else:
            index = 1
            self.file_view2.setText(file_name)

    def plot_maps(self):
        file_name1 = self.file_view1.text()
        file_name2 = self.file_view2.text()
        x_min = float(self.x_min.text())
        x_max = float(self.x_max.text())
        y_min = float(self.y_min.text())
        y_max = float(self.y_max.text())
        self.canvas.plot_maps_from_files(file_name1, file_name2, extent=[x_min, x_max, y_min, y_max])

    def recalculate_precursor(self):
        params_dict = {field.objectName(): float(field.text()) for fields in self.input_fields for field in fields}
        self.model.flush_results()
        self.model.set_experiment_data(**params_dict)
        results, comments = self.model.get_params(comments=True)
        results = keys_to_string(results)
        for items in self.result_fields:
            text = results[items.objectName()]
            try:
                if 1e-3 < abs(text) < 1e3:
                    items.setText(f'{text:.4f}')
                else:
                    items.setText(f'{text:.4e}')
            except TypeError:
                items.setText(str(text))
        self.comments_view.setText(comments)


if __name__ == "__main__":
    qApp = QApplication(sys.argv)

    aw = ApplicationWindow()
    aw.show()

    sys.exit(qApp.exec_())
