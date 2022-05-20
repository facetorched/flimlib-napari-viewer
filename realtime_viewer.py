import colorsys
import copy
import dataclasses
import json
import logging
import os
import sys
import time

import flimlib
import h5py
import napari
from napari.layers.shapes.shapes import Shapes
from napari.layers.points.points import Points
import numpy as np
from plot_widget import Fig

from dataclasses import dataclass
from magicgui import magicgui
from napari.qt.threading import thread_worker
from scipy.spatial import KDTree
from vispy.scene.visuals import Text

# need these for file dialogue?
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QWidget
#from PyQt5 import QtGui

from functools import wraps
from time import time

# copied from stackoverflow.com :)
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'Function {f.__name__} took {te-ts:2.4f} seconds')
        return result
    return wrap

FONT_SIZE = 10
PHASOR_SCALE = 1000
PHASOR_OPACITY_FACTOR = 0.2

# not actually empty. Ideally I could use None as input to napari but it doesn't like it
EMPTY_RGB_IMAGE = np.zeros((1,1,3))
EMPTY_PHASOR_IMAGE = np.zeros((1,3))
DEFAULT_POINT = np.zeros((1,2))
DEFAULT_PHASOR_POINT = np.array([[PHASOR_SCALE//2, PHASOR_SCALE//2]])
ELLIPSE = np.array([[59, 222], [110, 289], [170, 243], [119, 176]])
DEFUALT_MIN_INTENSITY = 10
DEFUALT_MAX_CHISQ = 200.0
NUM_PHASOR_BASE_LAYERS = 2
NUM_LIFETIME_BASE_LAYERS = 1
COLORMAP = np.array([colorsys.hsv_to_rgb(f, 1.0, 1) for f in np.linspace(0,1,256)])

class ListArray:
    """An array-like object backed by a list of ndarrays."""

    def __init__(self, arrays):
        if not len(arrays):
            raise ValueError # At least for now, don't allow empty

        self._arrays = []
        self._dtype = None
        self._shape = None
        for a in arrays:
            self._arrays.append(np.asarray(a))
            if self._dtype is None:
                self._dtype = self._arrays[0].dtype
            elif self._arrays[-1].dtype != self._dtype:
                raise ValueError
            if self._shape is None:
                self._shape = self._arrays[0].shape
            elif self._arrays[-1].shape != self._shape:
                raise ValueError

    @property
    def ndim(self):
        return 1 + len(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return (len(self._arrays),) + self._shape

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        ndslices = slices[1:]
        s0 = slices[0]
        if isinstance(s0, slice):
            start, stop, step = s0.indices(len(self._arrays))
            dim0 = (stop - start) // step
            shape = (dim0,) + self._shape
            ret = np.empty_like(self._arrays[0], shape=shape)
            j0 = 0
            for i0 in range(start, stop, step):
                ret[j0] = self._arrays[i0][ndslices]
                j0 += 1
            return ret
        else:  # s0 is an integer
            return self._arrays[s0][ndslices]

    def __array__(self):
        return self[:]
    
    def __len__(self):
        return np.prod(self.shape)

@dataclass
class SnapshotData:
    photon_count : np.ndarray
    phasor : np.ndarray
    phasor_quadtree : KDTree

@dataclass
class UserParameters:
    # Warning, members must not be mutable since copies of this class are made
    period : float
    fit_start : int
    fit_end : int
    min_intensity : int
    max_chisq : float
    max_tau : float
    is_snapshot_frames : bool

class SeriesViewer():

    def __init__(self, period=.04, fit_start=None, fit_end=None):
        # user parameters
        self.params = UserParameters(
            period = period,
            fit_start=fit_start,
            fit_end=fit_end,
            min_intensity=DEFUALT_MIN_INTENSITY,
            max_chisq=DEFUALT_MAX_CHISQ,
            max_tau=None,
            is_snapshot_frames = False,
        )

        # class members
        self.is_compute_thread_running = False
        self.lifetime_viewer = napari.Viewer(title="Lifetime Viewer")
        self.phasor_viewer = napari.Viewer(title="Phasor Viewer") #number this if there's more than one viewer?
        autoscale_viewer(self.phasor_viewer, (PHASOR_SCALE, PHASOR_SCALE))
        self.options_widget = None
        self.snapshots = []
        
        self.phasor_image_data = [None,]
        self.phasor_image = None
        self.lifetime_image_data = [None,]
        self.lifetime_image = None
        
        @self.lifetime_viewer.dims.events.current_step.connect
        def lifetime_slider_changed(event):
            if self.snapshots:
                self.phasor_viewer.dims.current_step = self.lifetime_viewer.dims.current_step
                self.update()

        # TODO remove this!!! (phasor should not have steps)
        @self.phasor_viewer.dims.events.current_step.connect
        def phasor_slider_changed(event):
            if self.snapshots:
                self.lifetime_viewer.dims.current_step = self.phasor_viewer.dims.current_step
                self.update()

        #add the phasor circle
        phasor_circle = np.asarray([[PHASOR_SCALE, 0.5 * PHASOR_SCALE],[0.5 * PHASOR_SCALE,0.5 * PHASOR_SCALE]])
        x_axis_line = np.asarray([[PHASOR_SCALE,0],[PHASOR_SCALE,PHASOR_SCALE]])
        phasor_shapes_layer = self.phasor_viewer.add_shapes([phasor_circle, x_axis_line], shape_type=['ellipse','line'], face_color='',)
        phasor_shapes_layer.editable = False
        
        self.colors = color_gen()

        #create dummy image layers
        self.phasor_image = self.phasor_viewer.add_points(EMPTY_PHASOR_IMAGE, name="Phasor", edge_width=0, size=3)
        self.phasor_image.editable = False
        self.lifetime_image = self.lifetime_viewer.add_image(EMPTY_RGB_IMAGE, rgb=True, name="Lifetime")

        #set up select layers
        create_lifetime_select_layer(self.lifetime_viewer, self.phasor_viewer, self, color=next(self.colors))
        #create_phasor_select_layer(self.phasor_viewer, self.lifetime_viewer, self, color=next(self.colors))
        
        self.create_add_selection_widget()
        self.create_options_widget()
        self.create_snap_widget()
        self.reset_current_step()

    def get_current_step(self):
        return self.lifetime_viewer.dims.current_step[0]

    def reset_current_step(self):
        self.lifetime_viewer.dims.set_current_step(0,0)
        self.phasor_viewer.dims.set_current_step(0,0)
    
    def get_tau_axis_size(self):
        return self.snapshots[0].photon_count.shape[-1]

    def get_image_shape(self):
        """returns shape of the image (height, width)"""
        return self.snapshots[0].photon_count.shape[-3:-1]

    def has_data(self):
        return bool(self.snapshots)

    @timing
    def setup(self):
        """
        setup occurs after the first frame has arrived. 
        At this point we know what shape the incoming data is
        """
        tau_axis_size = self.get_tau_axis_size()
        summed_photon_count = self.snapshots[0].photon_count.reshape(-1, tau_axis_size).sum(axis=0)
        
        if self.params.fit_start is None:
            self.params.fit_start = int(np.argmax(summed_photon_count)) #estimate fit start as the max in the data
        if self.params.fit_end is None:
            self.params.fit_end = int(np.max(np.nonzero(summed_photon_count)) + 1) # estimate fit end as bounding the last nonzero data
        self.params.max_tau = tau_axis_size * self.params.period # default is the width of the histogram
        params_copy = copy.deepcopy(self.params)
        self.setup_options_widget()
        self.update_options_widget(params_copy)
        autoscale_viewer(self.lifetime_viewer, self.get_image_shape())

    # called after new data arrives
    def receive_and_update(self, photon_count):
        # for now, we ignore all but the first channel
        photon_count = photon_count[tuple([0] * (photon_count.ndim - 3))]
        # check if this is the first time receiving data
        if not self.has_data():
            self.snapshots += [SnapshotData(photon_count,None,None)]
            self.setup()
        else:
            self.snapshots[-1].photon_count = photon_count # live frame is the last
        self.update()

    def update(self):
        if not self.is_compute_thread_running:
            self.is_compute_thread_running = True
            # Note: the contents of the passed objects are not modified in main thread
            step = self.get_current_step()
            # if these snapshots are frames, we want to subtract the previous snapshot
            if self.params.is_snapshot_frames and step > 0:
                photon_count = self.snapshots[step].photon_count - self.snapshots[step - 1].photon_count
            else:
                photon_count = self.snapshots[step].photon_count
            # make a copy to maintain thread safety
            params_copy = copy.deepcopy(self.params)
            worker = compute(photon_count, params_copy, step)
            worker.returned.connect(self.update_displays)
            worker.start()

    @timing
    def update_displays(self, arg):
        # this function is where compute thread returns to display thread
        self.is_compute_thread_running = False
        phasor, phasor_quadtree, phasor_image, phasor_intensity, lifetime_image, step = arg
        self.snapshots[step].phasor = phasor
        self.snapshots[step].phasor_quadtree = phasor_quadtree

        self.lifetime_image_data[step] = lifetime_image
        self.phasor_image_data[step] = phasor_image

        self.lifetime_image.data = ListArray(self.lifetime_image_data)
        # TODO make this memory efficient
        #set_points(self.phasor_image, ListArray(self.phasor_image_data), intensity=phasor_intensity)
        #self.phasor_image.data = np.array(self.phasor_image_data).reshape(-1, 3)
        set_points(self.phasor_image, np.array(self.phasor_image_data).reshape(-1, 3), intensity=phasor_intensity)
        self.phasor_image.editable = False
        self.update_selections()

    # TODO selections should only select the current viewed channel (if channels are added)
    @timing
    def update_selections(self):
        for layer in self.lifetime_viewer.layers:
            if 'selection' in layer.metadata:
                layer.metadata['selection'].update_co_selection()
        for layer in self.phasor_viewer.layers:
            if 'selection' in layer.metadata:
                layer.metadata['selection'].update_co_selection()
    # TODO save json with user settings
    def create_add_selection_widget(self):
        @magicgui(call_button="add lifetime selection")
        def add_lifetime_selection():
            create_lifetime_select_layer(self.lifetime_viewer, self.phasor_viewer, self, color=next(self.colors))

        @magicgui(call_button="add phasor selection")
        def add_phasor_selection():
            create_phasor_select_layer(self.phasor_viewer, self.lifetime_viewer, self, color=next(self.colors))

        add_lifetime_dw = self.lifetime_viewer.window.add_dock_widget(add_lifetime_selection, name='add lifetime selection', area='left')
        add_phasor_dw = self.phasor_viewer.window.add_dock_widget(add_phasor_selection, name = 'add phasor selection', area='left')

        # TODO remove access to private member
        add_lifetime_dw._close_btn = False
        add_phasor_dw._close_btn = False

    def create_snap_widget(self):
        @magicgui(call_button="snap")
        def snap():
            if self.has_data():
                prev = self.snapshots[-1]
                self.snapshots += [SnapshotData(prev.photon_count, prev.phasor, prev.phasor_quadtree)]
                self.lifetime_image_data += [self.lifetime_image_data[-1]]
                self.phasor_image_data += [self.phasor_image_data[-1]]
        self.lifetime_viewer.window.add_dock_widget(snap, name='snapshot', area='left')

    def create_options_widget(self):
        @magicgui(auto_call=True, 
            period={"label" : "Period (ns)"},
            fit_start={"label": "Fit Start"}, 
            fit_end={"label": "Fit End"},
            min_intensity={"label": "Min Intensity"},
            max_chisq={"label": "Max χ2"},
            max_tau={"label": "Max Lifetime"},
            is_snapshot_frames={"label": "Snapshots are frames"}
            )
        def options_widget(
            period : float = self.params.period,
            fit_start : int = self.params.fit_start,
            fit_end : int = self.params.fit_end,
            min_intensity : int = self.params.min_intensity,
            max_chisq : float = self.params.max_chisq,
            max_tau : float = self.params.max_tau,
            is_snapshot_frames : bool = self.params.is_snapshot_frames,
        ):
            self.params.period = period
            self.params.fit_start = fit_start
            self.params.fit_end = fit_end
            self.params.min_intensity = min_intensity
            self.params.max_chisq = max_chisq
            self.params.max_tau = max_tau
            self.params.is_snapshot_frames = is_snapshot_frames
            if self.has_data:
                self.update()
        
        self.lifetime_viewer.window.add_dock_widget(options_widget, name ='parameters', area='right')
        self.options_widget = options_widget

        @options_widget.fit_start.changed.connect
        def change_start_label(event):
            options_widget.fit_start.label = "Fit Start= {:.2f}ns".format(event * self.params.period)
        @options_widget.fit_end.changed.connect
        def change_end_label(event):
            options_widget.fit_end.label = "Fit End= {:.2f}ns".format(event * self.params.period)

        class FileSelector(QWidget):
            def save_file(self):
                filepath = QFileDialog.getSaveFileName(self, "Save Parameters", "./untitled.json", "Json (*.json)")
                save_widget.filepath.value = filepath[0] # set the value in the widget
                return filepath[0]
            def load_file(self):
                filepath = QFileDialog.getOpenFileName(self, "Load Parameters", "./untitled.json", "Json (*.json)")
                save_widget.filepath.value = filepath[0] # set the value in the widget
                return filepath[0]

        # TODO https://napari.org/magicgui/usage/_autosummary/magicgui.widgets.FileEdit.html

        @magicgui(
            call_button="save",
            filepath={'label': 'filepath'},
            load={'widget_type' : 'PushButton', 'label' : 'load'}
        )
        def save_widget(filepath='', load=True):
            if not filepath:
                fs = FileSelector()
                # TODO does this freeze everything else until user selects something?
                filepath = fs.save_file()
            print('saving parameters to', os.path.abspath(filepath))
            with open(filepath,'w') as outfile:
                json.dump(dataclasses.asdict(self.params), outfile, indent=4)

        def load_callback():
            filepath = save_widget.filepath.value
            if not filepath:
                fs = FileSelector()
                # TODO does this freeze everything else until user selects something?
                filepath = fs.load_file()
            print('loading parameters from', os.path.abspath(filepath))
            with open(filepath,'r') as infile:
                params_dict = json.load(infile)
                # the following only works if UserParameters continues to have only simple types
                self.update_options_widget(UserParameters(**params_dict))

        save_widget.load.clicked.connect(load_callback)
        self.lifetime_viewer.window.add_dock_widget(save_widget, name='save parameters', area='right')
    
    def setup_options_widget(self):
        tau_axis_size = self.get_tau_axis_size()
        self.options_widget.fit_start.min = 0
        self.options_widget.fit_start.max = tau_axis_size - 1
        self.options_widget.fit_end.min = 1
        self.options_widget.fit_end.max = tau_axis_size

    def update_options_widget(self, params : UserParameters):
        self.options_widget.period.value = params.period
        self.options_widget.fit_start.value = params.fit_start
        self.options_widget.fit_end.value = params.fit_end
        self.options_widget.min_intensity.value = params.min_intensity
        self.options_widget.max_chisq.value = params.max_chisq
        self.options_widget.max_tau.value = params.max_tau
        self.options_widget.is_snapshot_frames.value = params.is_snapshot_frames


# about 0.004 seconds for 256x256x256 data
@timing
def compute_lifetime_image(tau, intensity):
    intensity *= 1.0/intensity.max()
    tau *= 255/np.nanmax(tau) # TODO uneven bin sizes. last bin is used only if intensity is max
    np.nan_to_num(tau, copy=False)
    tau = tau.astype(int)
    intensity_scaled_tau = COLORMAP[tau]
    intensity_scaled_tau[...,0] *= intensity
    intensity_scaled_tau[...,1] *= intensity
    intensity_scaled_tau[...,2] *= intensity
    return intensity_scaled_tau

def compute_phasor_image(phasor, intensity):
    return phasor.reshape(-1,phasor.shape[-1]), intensity.ravel() * PHASOR_OPACITY_FACTOR

@thread_worker
@timing
def compute(photon_count, params : UserParameters, step):
    period = params.period
    fit_start = params.fit_start
    fit_end = params.fit_end
    photon_count = np.asarray(photon_count, dtype=np.float32)
    intensity = photon_count.sum(axis=-1)
    # about 0.5 sec for 256x256x256 data
    phasor = flimlib.GCI_Phasor(period, photon_count, fit_start=fit_start, fit_end=fit_end, compute_fitted=False)
    #reshape to work well with mapping / creating the image
    #TODO can i have the last dimension be tuple? this would simplify indexing later
    phasor = np.round(np.dstack([np.full_like(phasor.v, step), (1 - phasor.v) * PHASOR_SCALE, phasor.u * PHASOR_SCALE])).astype(int)
    phasor_quadtree = KDTree(phasor.reshape(-1, phasor.shape[-1])[:,1:])

    # about 0.1 sec for 256x256x256 data
    rld = flimlib.GCI_triple_integral_fitting_engine(period, photon_count, fit_start=fit_start, fit_end=fit_end, compute_residuals=False)
    tau = rld.tau

    tau[intensity < params.min_intensity] = np.nan
    tau[rld.chisq > params.max_chisq] = np.nan
    # negative lifetimes are not valid
    tau[tau<0] = np.nan 
    tau[tau > params.max_tau] = np.nan

    lifetime_image = compute_lifetime_image(tau, intensity)
    phasor_image, phasor_intensity = compute_phasor_image(phasor, intensity)

    return phasor, phasor_quadtree, phasor_image, phasor_intensity, lifetime_image, step

def compute_fits(photon_count, params : UserParameters):
    period = params.period
    fit_start = params.fit_start
    fit_end = params.fit_end
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
        # TODO remove access to private member
        self.dock_widget._close_btn = False
        # TODO float button crashes the entire app. Couldn't find a way to remove it. Fix the crash?
        
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
    
    def update_with_selection(self, selection, params : UserParameters):
        rld_selected, lm_selected = compute_fits(selection, params)
        period = params.period
        fit_start = params.fit_start
        fit_end = params.fit_end
        time = np.linspace(0, lm_selected.fitted.size * params.period, lm_selected.fitted.size, endpoint=False, dtype=np.float32)
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

class LifetimeSelectionMetadata():
    def __init__(self, selection : Shapes, co_selection : Points, decay_plot : CurveFittingPlot, series_viewer : SeriesViewer):
        self.selection = selection
        self.co_selection = co_selection
        self.decay_plot = decay_plot
        self.series_viewer = series_viewer
        
    def update_co_selection(self):
        points = get_bounded_points(self.selection, self.series_viewer.lifetime_image_data[0].shape[:2])
        # is it possible that the current step changes during this update computation?
        step = self.series_viewer.get_current_step()
        if(len(points) > 0):
            points_indexer = tuple(np.asarray(points).T)
            set_points(self.co_selection, self.series_viewer.snapshots[step].phasor[points_indexer][:,1:])
            histogram = np.mean(self.series_viewer.snapshots[step].photon_count[points_indexer],axis=0)
            self.update_decay_plot(histogram)
        else:
            empty_histogram = np.zeros(self.series_viewer.get_tau_axis_size()) + np.nan
            self.update_decay_plot(empty_histogram)
            set_points(self.co_selection, None)
        self.co_selection.editable = False

    def update_decay_plot(self, selection):
        self.decay_plot.update_with_selection(selection, self.series_viewer.params)

# copied code from above with different coselection updating
class PhasorSelectionMetadata():
    def __init__(self, selection : Shapes, co_selection : Points, decay_plot : CurveFittingPlot, series_viewer : SeriesViewer):
        self.selection = selection
        self.co_selection = co_selection
        self.decay_plot = decay_plot
        self.series_viewer = series_viewer

    def update_co_selection(self):
        step = self.series_viewer.get_current_step()
        if len(self.selection.data) == 0:
            empty_histogram = np.zeros(self.series_viewer.get_tau_axis_size()) + np.nan
            self.update_decay_plot(empty_histogram)
            set_points(self.co_selection, None)
            return
        extrema = np.ceil(self.selection._extent_data).astype(int) # verticies of bounding box [[x1, y1], [x2, y2]] where p1 < p2
        bounding_center = np.mean(extrema, axis=0)
        bounding_shape = extrema[1] - extrema[0] + np.ones(2, dtype=int) # add one since extremas are inclusive. does this make sense?
        bounding_radius = np.max(bounding_center - extrema[0]) # distance in the p = inf norm
        height, width = self.series_viewer.get_image_shape()
        maxpoints = width * height
        distances, indices = self.series_viewer.snapshots[step].phasor_quadtree.query(bounding_center, maxpoints, p=np.inf, distance_upper_bound=bounding_radius)
        n_indices = np.searchsorted(distances, np.inf)
        if n_indices > 0:
            indices = indices[0:n_indices]
            bounded_points = np.asarray([indices // height, indices % width]).T

            points = []
            offset=extrema[0]
            # use of private field `_data_view` since the shapes.py `to_masks()` fails to recognize offset
            mask = self.selection._data_view.to_masks(mask_shape=bounding_shape, offset=offset).astype(bool)[0]

            for point in bounded_points:
                bounded_phasor = self.series_viewer.snapshots[step].phasor[tuple(point)][1:] # phasor coordinates
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
                histogram = np.mean(self.series_viewer.snapshots[step].photon_count[points_indexer], axis=0)
                self.update_decay_plot(histogram)
        else:
            empty_histogram = np.zeros(self.series_viewer.get_tau_axis_size()) + np.nan
            self.update_decay_plot(empty_histogram)
            set_points(self.co_selection, None)
        self.co_selection.editable = False
        

    def update_decay_plot(self, selection):
        self.decay_plot.update_with_selection(selection, self.series_viewer.params)

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
    #except ValueError:
    #    print(points.shape)
    #    print(points_layer.data.shape)
    if intensity is not None:
        color = np.lib.stride_tricks.as_strided([1.0], shape=(points.shape[0],), strides=(0,))
        # TODO using resize to repeat the sequence is a terrible temporary solution
        points_layer.face_color = np.array([color,color,color,np.resize(intensity, points.shape[0])]).T
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
    co_viewer.layers.select_previous()
    co_viewer.layers.move(len(co_viewer.layers)-1, NUM_PHASOR_BASE_LAYERS)
    co_selection.editable = False
    decay_plot = CurveFittingPlot(viewer, scatter_color=color)
    selection.metadata = {'selection': LifetimeSelectionMetadata(selection, co_selection, decay_plot, series_viewer)}
    selection.mouse_drag_callbacks.append(select_shape_drag)
    selection.events.data.connect(handle_new_shape)
    selection.mode = 'select'
    return selection

# TODO according to jenu, selection in phasor and add button isn't needed. phasor should just be viewer
def create_phasor_select_layer(viewer, co_viewer, series_viewer, color='#FF0000'):
    selection = viewer.add_shapes(ELLIPSE, shape_type='ellipse', name="Selection", face_color=color+"7f", edge_width=0)
    co_selection = co_viewer.add_points(None, name="Correlation", size=1, face_color=color, edge_width=0)
    co_viewer.layers.select_previous()
    co_viewer.layers.move(len(co_viewer.layers)-1, NUM_LIFETIME_BASE_LAYERS)
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

