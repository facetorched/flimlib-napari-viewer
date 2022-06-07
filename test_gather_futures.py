from distutils.log import error
import threading
from time import sleep
import unittest

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future

MARGIN = .1
LONG_TIME = .5

def gather_futures(*futures):
    """
    Return a new future that completes when all the given futures complete.

    The value of the returned future is the tuple containing the values of all
    the given futures.

    If any of the futures complete with an exception, the returned future also
    completes with an exception (which is arbitrarily chosen among the given
    futures' exceptions).

    If the returned future is canceled, it has no effect on the given futures
    themselves.
    """

    ret = Future()

    # Immediately mark as running, because we may finish upon calling
    # add_done_callback()
    not_canceled = ret.set_running_or_notify_cancel()
    assert not_canceled

    # We need a reentrant lock because done_callback() may be called
    # synchronously inside add_done_callback()
    lock = threading.RLock() 

    with lock:
        unfinished = set(futures)
        if len(unfinished) < len(futures):
            raise ValueError("Futures must be distinct")

        results = [None] * len(futures)
        finished = [False]

        for i, fut in enumerate(futures):
            def done_callback(f):
                finished_results = None
                finished_exception = None
                with lock:
                    if finished[0]:
                        return
                    unfinished.remove(f)
                    try:
                        results[i] = f.result()
                        if not unfinished:
                            finished_results = tuple(results)
                            finished[0] = True
                    except Exception as e:
                        finished_exception = e
                        finished[0] = True
                    need_to_set_result = finished[0]

                if need_to_set_result:
                    if finished_results:
                        ret.set_result(finished_results)
                    else:
                        ret.set_exception(finished_exception)

            fut.add_done_callback(done_callback)

    return ret


executor = ThreadPoolExecutor(max_workers=1)
done_counter = 0

def fast_task():
    return

def slow_task():
    sleep(LONG_TIME)

def fast_error_task():
    raise RuntimeError("daga kotowaru")

def slow_error_task():
    sleep(LONG_TIME)
    raise RuntimeError("daga kotowaru")

def reset_done_counter():
    global done_counter
    done_counter = 0

def update_done_counter(f):
    global done_counter
    done_counter = True

class TestGatherFutures(unittest.TestCase):
    def test_one_fast(self):
        reset_done_counter()
        f1 = executor.submit(fast_task)
        all_done = gather_futures(f1)
        all_done.add_done_callback(update_done_counter)
        sleep(MARGIN)
        self.assertEqual(done_counter, 1)

    def test_many_fast(self):
        reset_done_counter()
        num = 10
        tasks = []
        for i in range(num):
            tasks += [executor.submit(fast_task)]
        all_done = gather_futures(*tasks)
        all_done.add_done_callback(update_done_counter)
        sleep(MARGIN * num)
        self.assertEqual(done_counter, 1)

    def test_one_slow(self):
        reset_done_counter()
        f1 = executor.submit(slow_task)
        all_done = gather_futures(f1)
        all_done.add_done_callback(update_done_counter)
        sleep(LONG_TIME + MARGIN)
        self.assertEqual(done_counter, 1)

    def test_two_slow(self):
        reset_done_counter()
        f1 = executor.submit(slow_task)
        f2 = executor.submit(slow_task)
        all_done = gather_futures(f1, f2)
        all_done.add_done_callback(update_done_counter)
        sleep((LONG_TIME + MARGIN) * 2)
        self.assertEqual(done_counter, 1)

    def test_one_fast_one_slow(self):
        reset_done_counter()
        f1 = executor.submit(fast_task)
        f2 = executor.submit(slow_task)
        all_done = gather_futures(f1, f2)
        all_done.add_done_callback(update_done_counter)
        sleep(LONG_TIME + MARGIN * 2)
        self.assertEqual(done_counter, 1)

    def test_one_slow_one_fast(self):
        reset_done_counter()
        f1 = executor.submit(slow_task)
        f2 = executor.submit(fast_task)
        all_done = gather_futures(f1, f2)
        all_done.add_done_callback(update_done_counter)
        sleep(LONG_TIME + MARGIN * 2)
        self.assertEqual(done_counter, 1)

    def test_fast_error(self):
        reset_done_counter()
        f1 = executor.submit(fast_error_task)
        all_done = gather_futures(f1)
        all_done.add_done_callback(update_done_counter)
        sleep(MARGIN)
        self.assertRaises(RuntimeError, all_done.result)
        self.assertEqual(done_counter, 1)

    def test_slow_error(self):
        reset_done_counter()
        f1 = executor.submit(fast_error_task)
        all_done = gather_futures(f1)
        all_done.add_done_callback(update_done_counter)
        sleep(LONG_TIME + MARGIN)
        self.assertRaises(RuntimeError, all_done.result)
        self.assertEqual(done_counter, 1)

if __name__ == "__main__":
    unittest.main()
