
import logging
import sys

import napari
import numpy as np
from napari.qt.threading import thread_worker

import flimstream
from realtime_viewer import SeriesViewer

@thread_worker
def receive(receiver):
    series_no = -1
    while True:
        series_receiver = receiver.wait_and_receive_series()
        if not series_receiver or series_receiver is Ellipsis:
            logging.info("Worker exiting")
            return
        series_no += 1
        while True:
            seqno, frame = series_receiver.wait_and_receive_element()
            if frame is None:
                break
            if frame is Ellipsis:
                logging.info("Worker exiting")
                return
            
            yield series_no, seqno, frame


def run(port):
    series_viewer = SeriesViewer()
    receiver = flimstream.Receiver(port)

    def element_callback(arg):
        nonlocal series_viewer
        series_no, seqno, frame = arg
        #TODO use series_no for naming viewers
        if series_no != 0 and seqno == 0:
            series_viewer = SeriesViewer()
        series_viewer.receive_and_update(frame)
        
    worker = receive(receiver)
    worker.quit = lambda: receiver.quit()
    worker.yielded.connect(element_callback)
    worker.start()
    
    logging.info("Starting napari")
    napari.run()
    logging.info("napari exited")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        _, port = sys.argv
        port = int(port)
    except:
        print("usage: python display-intensity.py port", file=sys.stderr)
        sys.exit(1)
    run(port)