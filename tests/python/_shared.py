import numpy as np


_shared_times = np.hstack(np.asarray(np.arange(10,50,1.5),
                                     np.arange(50, 1000, 15),
                                     np.arange(1000, 3100, 100)))


def times_no_irf():
    times = np.hstack((np.asarray(np.arange(0, 10, 0.01)),
                       _shared_times))
    return times


def times_with_irf():
    times = np.hstack((np.asarray(np.arange(-10, -1, 0.1),
                                  np.arange(-1, 10, 0.01)),
                       _shared_times))
    return times
