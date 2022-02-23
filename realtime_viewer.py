import colorsys
import logging
import sys
import time

import flimlib
import h5py
import napari
from napari.layers.shapes.shapes import Shapes
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
PHASOR_OPACITY_FACTOR = 0.2

# not actually empty. Ideally I could use None as input to napari but it doesn't like it
EMPTY_RGB_IMAGE = np.zeros((1,1,3))
DEFAULT_POINT = np.zeros((1,2))
DEFAULT_PHASOR_POINT = np.array([[PHASOR_SCALE//2, PHASOR_SCALE//2]])
ELLIPSE = np.array([[59, 222], [110, 289], [170, 243], [119, 176]])
DEFUALT_MIN_INTENSITY = 10

class SeriesViewer():
    def __init__(self, period=.04, fit_start=None, fit_end=None):
        self.period = period
        self.fit_start=fit_start
        self.fit_end=fit_end
        self.min_intensity=DEFUALT_MIN_INTENSITY
        self.photon_count = None
        self.phasor = None
        self.phasor_quadtree = None
        self.is_compute_thread_running = False
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

    def update_displays(self, arg):
        phasor, phasor_quadtree, phasor_image, phasor_intensity, lifetime_image = arg
        self.is_compute_thread_running = False

        self.phasor = phasor
        self.phasor_quadtree = phasor_quadtree
        self.lifetime_image.data = lifetime_image
        set_points(self.phasor_image, phasor_image, intensity=phasor_intensity)
        self.phasor_image.editable = False
        self.update_selections()

    # called after new data arrives
    def receive_and_update(self, photon_count):
        # check if this is the first time receiving data
        do_setup = self.photon_count is None
        self.photon_count = photon_count
        if do_setup:
            if self.fit_start is None:
                self.fit_start = int(np.argmax(np.sum(photon_count, axis=(0,1)))) #estimate fit start
            if self.fit_end is None:
                self.fit_end = self.photon_count.shape[-1]
            autoscale_viewer(self.lifetime_viewer, self.photon_count.shape[0:2])
            self.create_options_widget()

        self.update()

    def update(self):
        if not self.is_compute_thread_running:
            self.is_compute_thread_running = True
            worker = compute(self.photon_count, self.period, self.fit_start, self.fit_end, self.min_intensity)
            worker.returned.connect(self.update_displays)
            worker.start()

    def update_selections(self):
        for layer in self.lifetime_viewer.layers:
            if 'selection' in layer.metadata:
                layer.metadata['selection'].update_co_selection()
        for layer in self.phasor_viewer.layers:
            if 'selection' in layer.metadata:
                layer.metadata['selection'].update_co_selection()

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
            start={"label": "Fit Start= {:.2f}ns".format(self.fit_start * self.period), "max": self.photon_count.shape[-1]}, 
            end={"label": "Fit End= {:.2f}ns".format(self.fit_end * self.period), "max": self.photon_count.shape[-1]},
            intensity={"label": "Min Intensity"},
            )
        def options_widget(
            start : int = self.fit_start,
            end : int = self.fit_end,
            intensity : int = DEFUALT_MIN_INTENSITY,
        ):
            self.fit_start = start
            self.fit_end = end
            self.min_intensity = intensity
            self.update()
        self.lifetime_viewer.window.add_dock_widget(options_widget, area='left')
        
        @options_widget.start.changed.connect
        def change_start_label(event):
            options_widget.start.label = "Fit Start= {:.2f}ns".format(event.value * self.period)
        @options_widget.end.changed.connect
        def change_end_label(event):
            options_widget.end.label = "Fit End= {:.2f}ns".format(event.value * self.period)
    
def compute_lifetime_image(tau, intensity):
    intensity *= 1.0/intensity.max()
    tau *= 1.0/np.nanmax(tau)
    intensity_scaled_tau = np.zeros([*tau.shape,3], dtype=float)

    # temporary colormap (mark suggested a dict map)
    for r in range(tau.shape[0]):
        for c in range(tau.shape[1]):
            if not np.isnan(tau[r][c]):
                intensity_scaled_tau[r][c] = colorsys.hsv_to_rgb(tau[r][c], 1.0, intensity[r][c])

    return intensity_scaled_tau

def compute_phasor_image(phasor, intensity):
    return phasor.reshape(-1,2), intensity.ravel() * PHASOR_OPACITY_FACTOR

@thread_worker
def compute(photon_count, period, fit_start, fit_end, min_intensity):
    photon_count = np.asarray(photon_count, dtype=np.float32)
    intensity = photon_count.sum(axis=-1)
    print(np.mean(intensity))
    phasor = flimlib.GCI_Phasor(period, photon_count, fit_start=fit_start, fit_end=fit_end)
    #reshape to work well with mapping / creating the image
    #TODO can i have the last dimension be tuple? this would simplify indexing later
    phasor = np.round(np.dstack([(1 - phasor.v) * PHASOR_SCALE, phasor.u * PHASOR_SCALE])).astype(int)
    phasor_quadtree = KDTree(phasor.reshape(-1, 2))

    rld = flimlib.GCI_triple_integral_fitting_engine(period, photon_count, fit_start=fit_start, fit_end=fit_end)
    tau = rld.tau
    tau[tau<0] = np.nan
    
    tau[intensity < min_intensity] = np.nan
    #intensity_mean = np.nanmean(intensity)
    #intensity_std = np.std(intensity)

    #tau[intensity < intensity_mean - intensity_std] = np.nan

    #median_chisq = np.nanmedian(rld.chisq)
    #tau[np.isnan(rld.chisq)] = np.nan
    #tau[rld.chisq > median_chisq * 2] = np.nan # chisq seems to be going down???

    # a longer lifetime than the fit range is probably not valid
    tau[tau>(fit_end-fit_start) * period] = np.nan # TODO how should the contrast limits be handled?
    

    lifetime_image = compute_lifetime_image(tau, intensity)
    phasor_image, phasor_intensity = compute_phasor_image(phasor, intensity)

    return phasor, phasor_quadtree, phasor_image, phasor_intensity, lifetime_image

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
        time = np.linspace(0, lm_selected.fitted.size * period, lm_selected.fitted.size, endpoint=False, dtype=np.float32)
        fit_time = time[fit_start:fit_end]
        self.lm_curve.set_data((fit_time, lm_selected.fitted[fit_start:fit_end]))
        self.rld_curve.set_data((fit_time, rld_selected.fitted[fit_start:fit_end]))
        self.data_scatter.set_data(np.array((time, selection)).T, size=3, edge_width=0, face_color=self.scatter_color)
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
        if len(self.selection.data) == 0:
            empty_histogram = np.zeros(self.series_viewer.photon_count.shape[-1]) + np.nan
            self.update_decay_plot(empty_histogram)
            set_points(self.co_selection, None)
            return
        extrema = np.ceil(self.selection._extent_data).astype(int) # verticies of bounding box [[x1, y1], [x2, y2]] where p1 < p2
        bounding_center = np.mean(extrema, axis=0)
        bounding_shape = extrema[1] - extrema[0] + np.ones(2, dtype=int) # add one since extremas are inclusive. does this make sense?
        bounding_radius = np.max(bounding_center - extrema[0]) # distance in the p = inf norm
        height, width, _ = self.series_viewer.phasor.shape
        maxpoints = width * height
        distances, indices = self.series_viewer.phasor_quadtree.query(bounding_center, maxpoints, p=np.inf, distance_upper_bound=bounding_radius)
        n_indices = np.searchsorted(distances, np.inf)
        if n_indices > 0:
            indices = indices[0:n_indices]
            bounded_points = np.asarray([indices // height, indices % width]).T

            points = []
            offset=extrema[0]
            # use of private field `_data_view` since the shapes.py `to_masks()` fails to recognize offset
            mask = self.selection._data_view.to_masks(mask_shape=bounding_shape, offset=offset).astype(bool)[0]

            for point in bounded_points:
                bounded_phasor = self.series_viewer.phasor[tuple(point)]
                mask_indexer = tuple(bounded_phasor - offset)
                # kd tree found a square bounding box. some of these points might be outside of the rectangular mask
                if mask_indexer[0] < 0 or mask_indexer[1] < 0 or mask_indexer[0] >= bounding_shape[0] or mask_indexer[1] >= bounding_shape[1]:
                    continue
                if mask[mask_indexer]:
                    points += [point]
            if points:
                points = np.asarray(points)
                if np.any(points < 0):
                    raise ValueError("Negative index encountered while indexing image layer. This is outside the image!")
                set_points(self.co_selection, points)
                points_indexer = tuple(points.T)
                histogram = np.mean(self.series_viewer.photon_count[points_indexer], axis=0)
                self.update_decay_plot(histogram)
        else:
            empty_histogram = np.zeros(self.series_viewer.photon_count.shape[-1]) + np.nan
            self.update_decay_plot(empty_histogram)
            set_points(self.co_selection, None)
        self.co_selection.editable = False
        

    def update_decay_plot(self, selection):
        self.decay_plot.update_with_selection(selection, self.series_viewer.period, self.series_viewer.fit_start, self.series_viewer.fit_end)

def get_points(layer: Shapes):
        return get_bounded_points(layer, None)

def get_bounded_points(layer: Shapes, image_shape):
        return np.asarray(np.where(layer.to_masks(image_shape).astype(bool)[0])).T

def set_points(points_layer, points, intensity=None):
    try:
        points_layer.data = points if points is None or len(points) else None
    except OverflowError:
        # there seems to be a bug in napari with an overflow error
        pass
    
    if intensity is not None:
        color = np.lib.stride_tricks.as_strided([1.0], shape=(points.shape[0],), strides=(0,))
        points_layer.face_color = np.array([color,color,color,intensity]).T
    points_layer.selected_data = {}

def select_shape_drag(layer, event):
    layer.metadata['selection'].update_co_selection()
    yield
    while event.type == 'mouse_move':
        layer.metadata['selection'].update_co_selection()
        yield

def handle_new_shape(event):
    event_layer = event._sources[0]

    # make sure to check if each of these operations has already been done since
    # changing the data triggers this event which may cause infinite recursion
    #if event_layer.data.shape[0] > 0 and event_layer.data.dtype != int:
    #    event_layer.data = np.round(event_layer.data).astype(int)
    if len(event_layer.data) > 1 and event_layer.editable:
        event_layer.data = event_layer.data[-1:]
    if len(event_layer.data) > 0:
        if('selection' in event_layer.metadata):
            event_layer.metadata['selection'].update_co_selection()


def create_lifetime_select_layer(viewer, co_viewer, series_viewer, color='#FF0000'):
    selection = viewer.add_shapes(ELLIPSE, shape_type='ellipse', name="Selection", face_color=color+"7f", edge_width=0)
    co_selection = co_viewer.add_points(None, name="Correlation", size=1, face_color=color, edge_width=0)
    co_selection.editable = False
    decay_plot = CurveFittingPlot(viewer, scatter_color=color)
    selection.metadata = {'selection': LifetimeSelectionMetadata(selection, co_selection, decay_plot, series_viewer)}
    selection.mouse_drag_callbacks.append(select_shape_drag)
    selection.events.data.connect(handle_new_shape)
    selection.mode = 'select'
    return selection

def create_phasor_select_layer(viewer, co_viewer, series_viewer, color='#FF0000'):
    selection = viewer.add_shapes(ELLIPSE, shape_type='ellipse', name="Selection", face_color=color+"7f", edge_width=0)
    co_selection = co_viewer.add_points(None, name="Correlation", size=1, face_color=color, edge_width=0)
    co_selection.editable = False
    decay_plot = CurveFittingPlot(viewer, scatter_color=color)
    selection.metadata = {'selection': PhasorSelectionMetadata(selection, co_selection, decay_plot, series_viewer)}
    selection.mouse_drag_callbacks.append(select_shape_drag)
    selection.events.data.connect(handle_new_shape)
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

