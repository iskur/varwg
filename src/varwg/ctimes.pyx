from cpython.datetime cimport import_datetime
import_datetime()

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cpython cimport datetime
from datetime import datetime as pydatetime

cdef double min_per_day = 24. * 60
cdef double sec_per_day = min_per_day * 60
cdef double msec_per_day = sec_per_day * 1e6


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[::1] datetime2doy(list dt):
    """ Extracts the day of year as a float from the given datetimes.
    """
    cdef unsigned int i, N = len(dt)
    cdef double[::1] doys = np.empty(N, dtype=float)
    cdef datetime.datetime sub_dt
    for i in range(N):
        sub_dt = dt[i]
        tt = sub_dt.timetuple()
        doys[i] = (<int>tt.tm_yday +
                   <int>tt.tm_hour / 24. +
                   <int>tt.tm_min / min_per_day +
                   <int>tt.tm_sec / sec_per_day +
                   <int>sub_dt.microsecond / msec_per_day)
    return doys


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef doy_distance(double doy0, double[::1] doys):
    cdef int t, T = len(doys)
    cdef double dist
    dists = np.empty(T, dtype=np.float64)
    cdef double[::1] dist_view = dists
    for t in prange(T, nogil=True, schedule="guided"):
        dist = (doy0 - doys[t]) % 365.
        if dist > (365. / 2):
            dist_view[t] = 365 - dist
        else:
            dist_view[t] = dist
    return dists
    
# def periodic_distance(x1, x2, period):
#     """

#     Examples
#     --------
#     >>> periodic_distance(0, 23, 24)
#     array(1.0)
#     """
#     period = float(period)
#     dist = (x1 - x2) % period
#     return np.where(dist > period / 2, period - dist, dist)
