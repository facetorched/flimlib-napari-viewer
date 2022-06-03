import colorsys
import copy
import dataclasses
import json
import logging
import os
import sys
import time
from typing import List

import flimlib
import h5py
from matplotlib.cbook import flatten
import napari
from napari.layers.shapes.shapes import Shapes
from napari.layers.points.points import Points
import numpy as np
from plot_widget import Fig

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from magicgui import magicgui
from napari.qt.threading import thread_worker
from scipy.spatial import KDTree
from superqt import ensure_main_thread
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
EMPTY_PHASOR_IMAGE = np.zeros((1,2))
DEFAULT_POINT = np.zeros((1,2))
DEFAULT_PHASOR_POINT = np.array([[PHASOR_SCALE//2, PHASOR_SCALE//2]])
ELLIPSE = np.array([[59, 222], [110, 289], [170, 243], [119, 176]])
DEFUALT_MIN_INTENSITY = 10
DEFUALT_MAX_CHISQ = 200.0
NUM_PHASOR_BASE_LAYERS = 2
NUM_LIFETIME_BASE_LAYERS = 1
COLOR_DEPTH = 256
COLORMAP = np.array([colorsys.hsv_to_rgb(f, 1.0, 1) for f in np.linspace(0,1,COLOR_DEPTH)], dtype=np.float32)

executor = ThreadPoolExecutor()

class ComputeTask:
    def __init__(self, photon_count, params, frame_number, series_viewer):
        self.frame_number = frame_number
        self.photon_count = photon_count
        self.params = params
        self.series_viewer = series_viewer
        self.started = False

        self.intensity = None
        self.lifetime_image = None
        self.phasor = None
        self.phasor_quadtree = None
        self.phasor_image = None
        self.phasor_face_color = None
        self.done = None

    @ensure_main_thread
    def start(self):
        if not self.all_started(): # don't start more than once
            assert self.intensity is None
            self.intensity = executor.submit(compute_intensity, self.photon_count)
            self.lifetime_image = executor.submit(compute_lifetime_image, self.photon_count, self.intensity, self.params)
            self.phasor = executor.submit(compute_phasor, self.photon_count, self.params)
            self.phasor_quadtree = executor.submit(compute_phasor_quadtree, self.phasor)
            self.phasor_image = executor.submit(compute_phasor_image, self.phasor)
            self.phasor_face_color = executor.submit(compute_phasor_face_color, self.intensity)
            self.done = executor.submit(self.all_done())
            self.done.add_done_callback(self.series_viewer.update_selections_callback)

    @ensure_main_thread
    def cancel(self):
        if self.all_started(): # if looking at an old snapshot
            self.intensity.cancel()
            self.lifetime_image.cancel()
            self.phasor.cancel()
            self.phasor_quadtree.cancel()
            self.phasor_image.cancel()
            self.phasor_face_color.cancel()
            self.done.cancel()
    
    def all_started(self):
        return self.done is not None
    
    def all_done(self):
        self.intensity.result()
        self.lifetime_image.result()
        self.phasor.result()
        self.phasor_quadtree.result()
        self.phasor_image.result()
        self.phasor_face_color.result()
        print("#########ALLDONE##############")
        return True


class FutureArray:
    """
    An array-like object backed by a future
    """

    def __init__(self, future_array : Future, shape, dtype=np.float32):
        self._future_array = future_array
        self._dtype = dtype
        self._shape = shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, slices):
        return self._future_array.result()[slices]

    def __array__(self):
        return self[:]
    
    def __len__(self):
        return np.prod(self.shape)

class LifetimeImageProxy:
    """
    An array-like object backed by the collection of lifetime_image tasks
    """

    def __init__(self, task_list : List[ComputeTask], task_shape, dtype=np.float32):
        if not len(task_list):
            raise ValueError # At least for now, don't allow empty

        self._arrays = task_list
        self._dtype = dtype
        self._task_shape = task_shape

    @property
    def ndim(self):
        return 1 + len(self._task_shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return (len(self._arrays),) + self._task_shape

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        ndslices = slices[1:]
        s0 = slices[0]
        if isinstance(s0, slice):
            start, stop, step = s0.indices(len(self._arrays))
            dim0 = (stop - start) // step
            shape = (dim0,) + self._task_shape
            ret = np.empty(shape, dtype=self._dtype)
            j0 = 0
            for i0 in range(start, stop, step):
                task = self._arrays[i0]
                task.start()
                ret[j0] = task.lifetime_image.result()[ndslices]
                j0 += 1
            return ret
        else:  # s0 is an integer
            task = self._arrays[s0]
            task.start()
            return task.lifetime_image.result()[ndslices]

    def __array__(self):
        print("################## NAPARI IS REQUESTING THE FULL ARRAY #####################")
        return self[:]
    
    def __len__(self):
        return np.prod(self.shape)

@dataclass
class SnapshotData:
    photon_count : np.ndarray
    tasks : ComputeTask

@dataclass
class UserParameters:
    # Warning, this class gets copied frequently to be used by threads
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

        @self.lifetime_viewer.dims.events.current_step.connect
        def lifetime_slider_changed(event):
            self.swap_phasor_proxy()
            self.update_selections()
        
        self.phasor_viewer = napari.Viewer(title="Phasor Viewer") #number this if there's more than one viewer?
        autoscale_viewer(self.phasor_viewer, (PHASOR_SCALE, PHASOR_SCALE))
        self.options_widget = None
        self.snapshots : List[SnapshotData]= []
        
        self.phasor_image_data = [None,]
        self.phasor_image = None
        self.lifetime_image_data = [None,]
        self.lifetime_image = None

        self.frame_number = -1

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

    @ensure_main_thread
    def process_live_data_callback(self, done):
        self.process_live_data()

    def process_live_data(self):
        if not self.is_live_updated():
            tasks_to_cancel = self.snapshots[-1].tasks
            self.snapshots[-1].tasks = ComputeTask(self.get_photon_count(-1), copy.deepcopy(self.params), self.frame_number, self)
            self.swap_proxy_arrays()
            if tasks_to_cancel is not None: # first frame has no previous tasks
                tasks_to_cancel.cancel()
        else:
            print("already???????????????????????")

    def process_all_snapshots(self):
        params_copy = copy.deepcopy(self.params)
        for i in range(len(self.snapshots) - 1):
            tasks_to_cancel = self.snapshots[i].tasks
            self.snapshots[i].tasks = ComputeTask(self.get_photon_count(i), params_copy, self.frame_number, self)
            tasks_to_cancel.cancel()
        self.process_live_data()

    def swap_proxy_arrays(self):
        task_list = self.get_tasks()
        self.lifetime_image.data = LifetimeImageProxy(task_list, self.get_image_shape() + (3,))
        self.swap_phasor_proxy()

    def swap_phasor_proxy(self):
        step = self.get_current_step()
        tasks = self.snapshots[step].tasks
        tasks.start()
        try:
            self.phasor_image.data = FutureArray(tasks.phasor_image, (self.get_num_phasors(),) + (2,), dtype=int)
        except OverflowError:
            print("##########WHYOVERFLOWS?###########")
        self.phasor_image.selected_data = {}
        self.phasor_image.face_color = compute_phasor_face_color(tasks.intensity) # fast enough for ui thread?
        #self.phasor_image.face_color = FutureArray(task_list[step].phasor_face_color, (self.get_num_phasors(),) + (4,))
        self.phasor_image.editable = False # need this?

    def get_tasks(self):
        return [snapshot.tasks for snapshot in self.snapshots]

    def is_live_updated(self):
        tasks = self.snapshots[-1].tasks
        return tasks is not None and tasks.frame_number == self.frame_number

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

    def get_num_phasors(self):
        return np.prod(self.get_image_shape())

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
        self.frame_number = self.frame_number + 1
        # for now, we ignore all but the first channel
        photon_count = photon_count[tuple([0] * (photon_count.ndim - 3))]
        # check if this is the first time receiving data
        if not self.has_data():
            self.snapshots += [SnapshotData(photon_count, None)]
            self.setup()
            self.process_live_data()
        else:
            self.snapshots[-1].photon_count = photon_count # live frame is the last
            tasks = self.snapshots[-1].tasks
            if tasks.all_started():
                tasks.done.add_done_callback(self.process_live_data_callback)
            else:
                # if we are not on the live layer
                self.process_live_data()

    def get_photon_count(self, step):
        if self.params.is_snapshot_frames and step != 0:
            photon_count = self.snapshots[step].photon_count - self.snapshots[step - 1].photon_count
        else:
            photon_count = self.snapshots[step].photon_count
        return photon_count

    @ensure_main_thread
    def update_selections_callback(self, done):
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
                self.snapshots += [SnapshotData(prev.photon_count, ComputeTask(prev.photon_count, copy.deepcopy(self.params), self.frame_number, self))]
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
                self.process_all_snapshots()
        
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
def compute_lifetime_image(photon_count, intensity_future : Future, params : UserParameters):
    period = params.period
    fit_start = params.fit_start
    fit_end = params.fit_end
    rld = flimlib.GCI_triple_integral_fitting_engine(period, photon_count, fit_start=fit_start, fit_end=fit_end, compute_residuals=False)
    tau = rld.tau
    
    intensity = intensity_future.result()
    tau[intensity < params.min_intensity] = np.nan
    tau[rld.chisq > params.max_chisq] = np.nan
    # negative lifetimes are not valid
    tau[tau<0] = np.nan 
    tau[tau > params.max_tau] = np.nan

    intensity = intensity / intensity.max()
    tau *= (COLOR_DEPTH)/np.nanmax(tau)
    np.nan_to_num(tau, copy=False)
    tau = tau.astype(int)
    tau[tau >= COLOR_DEPTH] = COLOR_DEPTH - 1 # this value is used to index into the colormap
    intensity_scaled_tau = COLORMAP[tau]
    intensity_scaled_tau[...,0] *= intensity
    intensity_scaled_tau[...,1] *= intensity
    intensity_scaled_tau[...,2] *= intensity
    return intensity_scaled_tau

def compute_intensity(photon_count : np.ndarray):
    return photon_count.sum(axis=-1)

def compute_phasor_image(phasor : Future):
    return phasor.result().reshape(-1,phasor.result().shape[-1])

def compute_phasor_face_color(intensity : Future):
    it = intensity.result()
    it = it / it.max()
    phasor_intensity = it.ravel() * PHASOR_OPACITY_FACTOR
    color = np.broadcast_to(1.0, phasor_intensity.shape)
    return np.asarray([color,color,color,phasor_intensity]).T

def compute_phasor(photon_count, params : UserParameters):
    period = params.period
    fit_start = params.fit_start
    fit_end = params.fit_end
    # about 0.5 sec for 256x256x256 data
    phasor = flimlib.GCI_Phasor(period, photon_count, fit_start=fit_start, fit_end=fit_end, compute_fitted=False)
    #reshape to work well with mapping / creating the image
    #TODO can i have the last dimension be tuple? this would simplify indexing later
    return np.round(np.dstack([(1 - phasor.v) * PHASOR_SCALE, phasor.u * PHASOR_SCALE])).astype(int)

def compute_phasor_quadtree(phasor : Future):
    return KDTree(phasor.result().reshape(-1, phasor.result().shape[-1]))

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
        step = self.series_viewer.get_current_step()
        tasks = self.series_viewer.snapshots[step].tasks
        if tasks.done.running():
            return
        points = get_bounded_points(self.selection, self.series_viewer.get_image_shape())
        # is it possible that the current step changes during this update computation?
        
        if(len(points) > 0):
            points_indexer = tuple(np.asarray(points).T)
            set_points(self.co_selection, tasks.phasor.result(timeout=0)[points_indexer])
            histogram = np.mean(self.series_viewer.get_photon_count(0)[points_indexer],axis=0)
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
        tasks = self.series_viewer.snapshots[step].tasks
        if tasks.done.running():
            return
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
        distances, indices = tasks.phasor_quadtree.result(timeout=0).query(bounding_center, maxpoints, p=np.inf, distance_upper_bound=bounding_radius)
        n_indices = np.searchsorted(distances, np.inf)
        if n_indices > 0:
            indices = indices[0:n_indices]
            bounded_points = np.asarray([indices // height, indices % width]).T

            points = []
            offset=extrema[0]
            # use of private field `_data_view` since the shapes.py `to_masks()` fails to recognize offset
            mask = self.selection._data_view.to_masks(mask_shape=bounding_shape, offset=offset).astype(bool)[0]

            for point in bounded_points:
                bounded_phasor = tasks.phasor.result(timeout=0)[tuple(point)]
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
                histogram = np.mean(self.series_viewer.get_photon_count()[points_indexer], axis=0)
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
        color = np.broadcast_to(1.0, (points.shape[0],))
        points_layer.face_color = np.asarray([color,color,color,intensity]).T
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

