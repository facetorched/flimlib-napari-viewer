import logging
import numpy as np
import sys
import time

import h5py

import flimstream


def get_h5_dset(filename, key):
    f = h5py.File(filename)
    return f[key]


def send_series(port, filename, interval):
    f = h5py.File(filename)
    keys= [key for key in f.keys()]

    first = f.get(keys[0])

    sender = flimstream.SeriesSender(first.dtype, first.shape, port)

    sender.start()

    for i, key in enumerate(keys):
        start = time.time()

        sender.send_element(i, f.get(key))

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
        _, port, filename, interval = sys.argv
        port = int(port)
        interval = float(interval)
    except:
        print("usage: python send-h5.py port filename interval",
                file=sys.stderr)
        sys.exit(1)
    send_series(port, filename, interval)
