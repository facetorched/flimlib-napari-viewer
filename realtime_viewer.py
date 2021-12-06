import colorsys
import logging
import sys
import time

import flimlib
import h5py
import napari
import numpy as np
from plot_widget import Fig

from magicgui import magicgui
from napari.qt.threading import thread_worker
from scipy.spatial import KDTree
from vispy.scene.visuals import Text

FONT_SIZE = 10
TAU_MAX = 4
CHISQ_MAX = 200
PHASOR_SCALE = 1000

# not actually empty. Ideally I could use None as input to napari but it doesn't like it
EMPTY_RGB_IMAGE = np.zeros((1,1,3))
DEFAULT_POINT = np.zeros((1,2))
DEFAULT_PHASOR_POINT = np.array([[PHASOR_SCALE//2, PHASOR_SCALE//2]])

class SeriesViewer():
    def __init__(self, period=.04, fit_start=None, fit_end=None):
        self.period = period
        self.fit_start=fit_start
        self.fit_end=fit_end
        self.photon_count = None
        self.phasor = None
        self.phasor_quadtree = None
        self.intensity = None
        self.tau = None
        self.phasor_viewer = napari.Viewer(title="Phasor Viewer") #number this if there's more than one viewer?
        self.lifetime_viewer = napari.Viewer(title="Lifetime Viewer")
        #napari.run()
        self.phasor_image = self.phasor_viewer.add_points(None, name="Phasor", edge_width=0, size = 3)
        self.phasor_image.editable = False
        self.lifetime_image = self.lifetime_viewer.add_image(EMPTY_RGB_IMAGE, rgb=True, name="Lifetime")
        
        autoscale_viewer(self.phasor_viewer, (PHASOR_SCALE, PHASOR_SCALE))
        #add the phasor circle
        phasor_circle = np.asarray([[PHASOR_SCALE, 0.5 * PHASOR_SCALE],[0.5 * PHASOR_SCALE,0.5 * PHASOR_SCALE]])
        x_axis_line = np.asarray([[PHASOR_SCALE,0],[PHASOR_SCALE,PHASOR_SCALE]])
        phasor_shapes_layer = self.phasor_viewer.add_shapes([phasor_circle, x_axis_line], shape_type=['ellipse','line'], face_color='',)
        phasor_shapes_layer.editable = False

        self.colors = color_gen()

        #set up select layers
        create_lifetime_select_layer(self.lifetime_viewer, self.phasor_viewer, self, color=next(self.colors))
        create_phasor_select_layer(self.phasor_viewer, self.lifetime_viewer, self, color=next(self.colors))
        
        self.create_add_selection_widget()

    def update_image_layers(self):
        self.update_lifetime_image(self.tau, self.intensity)
        self.update_phasor_image(self.phasor, self.intensity)

    def update_lifetime_image(self, tau, intensity):
        intensity *= 1.0/intensity.max()
        tau *= 1.0/np.nanmax(tau)
        intensity_scaled_tau = np.zeros([*tau.shape,3], dtype=float)

        # temporary colormap (mark suggested a dict map)
        for r in range(tau.shape[0]):
            for c in range(tau.shape[1]):
                if not np.isnan(tau[r][c]):
                    intensity_scaled_tau[r][c] = colorsys.hsv_to_rgb(tau[r][c], 1.0, intensity[r][c])

        self.lifetime_image.data = intensity_scaled_tau

    def update_phasor_image(self, phasor, intensity):
        set_points(self.phasor_image, phasor.reshape(-1,2), intensity=intensity.ravel() * .1)
        self.phasor_image.editable = False

    # called after new data arrives
    def receive_and_update(self, photon_count):
        # check if this is the first time receiving data
        do_setup = self.photon_count is None
        self.photon_count = np.asarray(photon_count, dtype=np.float32)
        if do_setup:
            if self.fit_start is None:
                self.fit_start = 0
            if self.fit_end is None:
                self.fit_end = self.photon_count.shape[-1]
            autoscale_viewer(self.lifetime_viewer, self.photon_count.shape[0:2])
            self.create_options_widget()
        self.update()

    # called after adjusting fit_start/fit_end
    def update(self):
        self.compute()
        self.update_image_layers()
        self.update_selections()

    def update_selections(self):
        for layer in self.lifetime_viewer.layers:
            if 'selection' in layer.metadata:
                layer.metadata['selection'].update_co_selection()
        for layer in self.phasor_viewer.layers:
            if 'selection' in layer.metadata:
                layer.metadata['selection'].update_co_selection()

    def compute(self):
        psr = flimlib.GCI_Phasor(self.period, self.photon_count, fit_start=self.fit_start, fit_end=self.fit_end)
        #reshape to work well with mapping / creating the image
        self.phasor = np.round(np.dstack([(psr.v * -1 + 1 ) * PHASOR_SCALE, psr.u * PHASOR_SCALE])).astype(int)
        self.phasor_quadtree = KDTree(self.phasor.reshape(-1, 2))

        rld = flimlib.GCI_triple_integral_fitting_engine(self.period, self.photon_count, fit_start=self.fit_start, fit_end=self.fit_end)
        tau = rld.tau
        tau[tau<0] = np.nan
        tau[tau>TAU_MAX] = np.nan # TODO how should the contrast limits be handled?
        tau[rld.chisq > CHISQ_MAX] = np.nan
        self.tau = tau
        self.intensity = self.photon_count.sum(axis=-1)

    def create_add_selection_widget(self):
        @magicgui(call_button="add lifetime selection")
        def add_lifetime_selection():
            create_lifetime_select_layer(self.lifetime_viewer, self.phasor_viewer, self, color=next(self.colors))

        @magicgui(call_button="add phasor selection")
        def add_phasor_selection():
            create_phasor_select_layer(self.phasor_viewer, self.lifetime_viewer, self, color=next(self.colors))

        self.lifetime_viewer.window.add_dock_widget(add_lifetime_selection, area='left')
        self.phasor_viewer.window.add_dock_widget(add_phasor_selection, area='left')

    def create_options_widget(self):
        @magicgui(auto_call=True, 
            start={"label": "Fit Start", "max": self.photon_count.shape[-1] * self.period}, 
            end={"label": "Fit End", "max": self.photon_count.shape[-1] * self.period}
            )
        def options_widget(
            start : float = self.fit_start * self.period,
            end : float = self.fit_end * self.period,
        ):
            self.fit_start = int(round(start/self.period))
            self.fit_end = int(round(end/self.period))
            self.update()
        self.lifetime_viewer.window.add_dock_widget(options_widget, area='left')
        
def compute_fits(photon_count, period, fit_start, fit_end):
    rld = flimlib.GCI_triple_integral_fitting_engine(period, photon_count, fit_start=fit_start, fit_end=fit_end)
    param_in = [rld.Z, rld.A, rld.tau]
    lm = flimlib.GCI_marquardt_fitting_engine(period, photon_count, param_in, fit_start=fit_start, fit_end=fit_end)
    return rld, lm

def autoscale_viewer(viewer, shape):
    state = {'rect': ((0, 0), shape)}
    viewer.window.qt_viewer.view.camera.set_state(state)

class CurveFittingPlot():
    #TODO add transform into log scale
    def __init__(self, viewer, scatter_color='magenta'):
        self.fig = Fig()
        # add a docked figure
        self.dock_widget = viewer.window.add_dock_widget(self.fig, area='bottom')
        # get a handle to the plotWidget
        self.ax = self.fig[0, 0]
        self.lm_curve = self.ax.plot(None, color='g', marker_size=0, width=2)
        self.rld_curve = self.ax.plot(None, color='r', marker_size=0, width=2)
        self.scatter_color = scatter_color
        self.data_scatter = self.ax.scatter(None, size=1, edge_width=0, face_color=scatter_color)
        self.fit_start_line = self.ax.plot(None, color='b', marker_size=0, width=2)
        self.fit_end_line = self.ax.plot(None, color='b', marker_size=0, width=2)
        self.rld_info = Text(None, parent=self.ax.view, color='r', anchor_x='right', font_size = FONT_SIZE)
        self.lm_info = Text(None, parent=self.ax.view, color='g', anchor_x='right', font_size = FONT_SIZE)
    
    def update_with_selection(self, selection, period, fit_start, fit_end):
        rld_selected, lm_selected = compute_fits(selection, period, fit_start, fit_end)
        lm_time = np.linspace(0,(lm_selected.fitted.size-1)*period,lm_selected.fitted.size,dtype=np.float32)
        rld_time = lm_time[0:rld_selected.fitted.size]
        self.lm_curve.set_data((lm_time, lm_selected.fitted))
        self.rld_curve.set_data((rld_time, rld_selected.fitted))
        self.data_scatter.set_data(np.array((lm_time, selection)).T, size=3, edge_width=0, face_color=self.scatter_color)
        
        self.rld_info.pos = self.ax.view.size[0], self.rld_info.font_size
        self.rld_info.text = 'RLD | chisq = ' + "{:.2e}".format(float(rld_selected.chisq)) + ', tau = ' + "{:.2e}".format(float(rld_selected.tau))
        self.lm_info.pos = self.ax.view.size[0], self.rld_info.font_size*3
        self.lm_info.text = 'LMA | chisq = ' + "{:.2e}".format(float(lm_selected.chisq)) + ', tau = ' + "{:.2e}".format(float(lm_selected.param[2]))
        
        # autoscale based on data (ignore start/end lines)
        self.fit_start_line.set_data(np.zeros((2,1)))
        self.fit_end_line.set_data(np.zeros((2,1)))
        self.ax.autoscale()
        self.fit_start_line.set_data(([fit_start * period, fit_start * period], self.ax.camera._ylim))
        self.fit_end_line.set_data(([fit_end * period, fit_end * period], self.ax.camera._ylim))

        #TODO figure out where this gets called. after moving/adding a point and also modifying fitstart/fit end

class LifetimeSelectionMetadata():
    def __init__(self, selection, co_selection, decay_plot, series_viewer):
        self.selection = selection
        self.co_selection = co_selection
        self.decay_plot = decay_plot
        self.series_viewer = series_viewer
        
    def update_co_selection(self):
        points = get_bounded_points(self.selection, self.series_viewer.phasor.shape[:2])
        
        if(len(points) > 0):
            points_indexer = tuple(np.asarray(points).T)
            set_points(self.co_selection, self.series_viewer.phasor[points_indexer])
            histogram = np.mean(self.series_viewer.photon_count[points_indexer],axis=0)
            self.update_decay_plot(histogram)
        else:
            empty_histogram = np.zeros(self.series_viewer.photon_count.shape[-1]) + np.nan
            self.update_decay_plot(empty_histogram)
            set_points(self.co_selection, None)
        self.co_selection.editable = False

    def update_decay_plot(self, selection):
        self.decay_plot.update_with_selection(selection, self.series_viewer.period, self.series_viewer.fit_start, self.series_viewer.fit_end)

# copied code from above with different coselection updating
class PhasorSelectionMetadata():
    def __init__(self, selection, co_selection, decay_plot, series_viewer):
        self.selection = selection
        self.co_selection = co_selection
        self.decay_plot = decay_plot
        self.series_viewer = series_viewer

    def update_co_selection(self):
        selection_radius = int(self.selection.current_size) - 1 # TODO: Use correct selection radius
        selection_center = self.selection.data[0]

        height, width, _ = self.series_viewer.phasor.shape
        maxpoints = width * height
        distances, indices = self.series_viewer.phasor_quadtree.query(selection_center, maxpoints, distance_upper_bound=selection_radius)
        n_indices = np.searchsorted(distances, np.inf)
        if n_indices > 0:
            indices = indices[0:n_indices]
            x, y = indices % width, indices // height 
            set_points(self.co_selection, np.column_stack((y, x)))
            histograms = self.series_viewer.photon_count[y, x]
            histogram = np.mean(histograms, axis=0)
            self.update_decay_plot(histogram)
        else:
            empty_histogram = np.zeros(self.series_viewer.photon_count.shape[-1]) + np.nan
            self.update_decay_plot(empty_histogram)
            set_points(self.co_selection, None)
        self.co_selection.editable = False
        

    def update_decay_plot(self, selection):
        self.decay_plot.update_with_selection(selection, self.series_viewer.period, self.series_viewer.fit_start, self.series_viewer.fit_end)

def get_points(layer):
        point_size = int(layer.current_size) - 1 #THIS IS NOT CORRECT (I have no idea what napari considers as point size)
        points = []      
        for point in layer.data.astype(int):
            for r in range(point[0] - point_size, point[0] + point_size + 1):
                for c in range(point[1] - point_size, point[1] + point_size + 1):
                    points += [[r, c]]
        return points

def get_bounded_points(layer, image_shape):
        points = []
        point_size = int(layer.current_size) - 1 #THIS IS NOT CORRECT (I have no idea what napari considers as point size)
        for point in layer.data.astype(int):
            rmax = image_shape[0] - 1
            cmax = image_shape[1] - 1
            for r in range(min(rmax, max(0, point[0] - point_size)), min(rmax, max(0, point[0] + point_size + 1))):
                for c in range(min(cmax, max(0, point[1] - point_size)), min(cmax, max(0, point[1] + point_size + 1))):
                    points += [[r, c]]
        return points

def set_points(points_layer, points, intensity=None):
    points_layer.data = points if points is None or len(points) else None
    
    if intensity is not None:
        color = np.lib.stride_tricks.as_strided([1.0], shape=(points.shape[0],), strides=(0,))
        points_layer.face_color = np.array([color,color,color,intensity]).T
    points_layer.selected_data = {}

def select_points_drag(layer, event):
    try:
        layer.metadata['selection'].update_co_selection()
        yield
        while event.type == 'mouse_move':
            layer.metadata['selection'].update_co_selection()
            yield
    except Exception as e:
        print("select_points_drag")
        print(event.type)
        print(e)

def handle_new_point(event):
    event_layer = event._sources[0]
    try:
        # make sure to check if each of these operations has already been done since
        # changing the data triggers this event which may cause infinite recursion
        if event_layer.data.shape[0] > 0 and event_layer.data.dtype != int:
            event_layer.data = np.round(event_layer.data).astype(int)
        if event_layer.data.shape[0] > 1 and event_layer.editable:
            event_layer.data = event_layer.data[-1:]
        if event_layer.data.shape[0] > 0:
            if('selection' in event_layer.metadata):
                event_layer.metadata['selection'].update_co_selection()
    except Exception as e:
        print("handle_new_point")
        print(e)

def create_lifetime_select_layer(viewer, co_viewer, series_viewer, color='#FF0000'):
    selection = viewer.add_points(DEFAULT_POINT, name="Selection", symbol="square", face_color=color+"7f", edge_width=0)
    co_selection = co_viewer.add_points(None, name="Correlation", size=1, face_color=color, edge_width=0)
    co_selection.editable = False
    decay_plot = CurveFittingPlot(viewer, scatter_color=color)
    selection.metadata = {'selection': LifetimeSelectionMetadata(selection, co_selection, decay_plot, series_viewer)}
    selection.mouse_drag_callbacks.append(select_points_drag)
    selection.events.data.connect(handle_new_point)
    selection.mode = 'select'
    return selection

def create_phasor_select_layer(viewer, co_viewer, series_viewer, color='#FF0000'):
    selection = viewer.add_points(DEFAULT_PHASOR_POINT, name="Selection", symbol="square", face_color=color+"7f", edge_width=0)
    co_selection = co_viewer.add_points(None, name="Correlation", size=1, face_color=color, edge_width=0)
    co_selection.editable = False
    decay_plot = CurveFittingPlot(viewer, scatter_color=color)
    selection.metadata = {'selection': PhasorSelectionMetadata(selection, co_selection, decay_plot, series_viewer)}
    selection.mouse_drag_callbacks.append(select_points_drag)
    selection.events.data.connect(handle_new_point)
    selection.mode = 'select'
    return selection

COLOR_DICT = {  'red':"#FF0000",
                'green':"#00FF00",
                'blue':"#0000FF",
                'cyan':"#00FFFF",
                'magenta':"#FF00FF",
                'yellow':"#FFFF00",
            }

def color_gen():
    while True:
        for i in COLOR_DICT.keys():
            yield COLOR_DICT[i]

