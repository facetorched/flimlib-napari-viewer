import logging
import numpy as np
import sys
import time

import h5py

import flimstream


def get_h5_dset(filename, key):
    f = h5py.File(filename)
    return f[key]


def send_series(port, filename, frames, interval):
    f = h5py.File(filename)
    keys= [key for key in f.keys()]
    keys= keys[:frames]

    frame = np.asarray(f.get(keys[0]), dtype=np.uint16)

    sender = flimstream.SeriesSender(frame.dtype, frame.shape, port)
    sender.start()

    for i, key in enumerate(keys):
        start = time.time()

        if(i != 0):
            frame += f.get(key)
        sender.send_element(i, frame)

        finish = time.time()
        excess = finish - start - interval
        if excess > 0:
            logging.warning(f"Frame interval exceeded by {excess} s")
        else:
            time.sleep(-excess)
    sender.end()

    # The tempdir will be deleted when sender is GC'd, so wait
    print("Press CTRL-C to exit", file=sys.stderr)
    try:
        time.sleep(1e6)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        _, port, filename, frames, interval = sys.argv
        port = int(port)
        frames = int(frames)
        interval = float(interval)
    except:
        print("usage: python send-h5.py port filename frames interval",
                file=sys.stderr)
        sys.exit(1)
    send_series(port, filename, frames, interval)
