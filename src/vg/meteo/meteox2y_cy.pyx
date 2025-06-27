# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from datetime import datetime

import numpy as np
from libc.math cimport sin, cos, pow
# from libc.stdlib cimport malloc
from cython.parallel import prange, parallel
from timezonefinder import TimezoneFinder
import pytz
import bottleneck

from vg.ctimes import datetime2doy

cimport numpy as np

ctypedef np.double_t DTYPE_t

tzf = TimezoneFinder()


def pot_s_rad(date, lat, longt, in_format='%Y-%m-%dT%H:%M', tz_mer=15.0,
            wog=-1):
    """ theoretical maximal potential solar radiation outside of atmosphere

    Parameters
    ----------
    date : numpy.array of time strings (format: in_format) or datetime objects
        or floats (doys)
    lat : float, optional
        latitude of station in decimal degrees, default: Stuttgart Lauchaecker
    longt : float, optional
        longitude of station in decimal degrees, default: Stuttgart Lauchaecker
    in_format : time format string, optional
        format of date if date is string, default '%Y-%m-%dT%H:%M'
    tz_mer : int, optional
        central meridian of time zone, default: 15 (CET)
    wog : {-1, 1}, optional
        west of greenwich, 1 if west, -1 if east, default = -1

    Returns
    -------
    smax : numpy.array of floats
        maximal potential solar radiation in W/m^2

    Notes
    -----
    from campbell technical note 18 [1]_, except declination of sun (d):
    formula of Spencer (1971) [2]_

    WARNING: Campbell Scientific recommends the use of a high quality sun
    screen lotion when exposing your skin to solar radiation for large values
    of sunshine hours!

    References
    ----------
    .. [1] Campbell Scientific (2005) technical note 18: CALCULATING SUNSHINE
       HOURS FROM PYRANOMETER / SOLARIMETER DATA
    .. [2] Spencer JW (1971) Fourier series representation of the position of
       the Sun. Search 2: 172.

    Examples
    --------
    >>> date_str = np.array(["2011-09-28T11:27"])
    >>> pot_s_rad(date_str)
    array([ 855.35624182])
    >>> from vg import times
    >>> dt = times.str2datetime(date_str, "%Y-%m-%dT%H:%M")
    >>> pot_s_rad(dt)
    array([ 855.35624182])
    >>> pot_s_rad(times.datetime2doy(dt))
    array([ 855.35624182])
    """
    cdef double j1, j2, Et, t0, la, d, sinphi
    cdef int i
    cdef int n = len(date)
    cdef double pi = np.pi
    cdef double[::] smax = np.empty_like(date, dtype=float)
    cdef int s0 = 1373  # Solarkonstante W/m^2
    if tz_mer is None:
        timezone_str = tzf.timezone_at(lat=float(lat),
                                       lng=float(longt))
        timezone = pytz.timezone(timezone_str)
        dt = datetime(2020, 6, 1)
        tz_offset = ((timezone.utcoffset(dt, is_dst=False).total_seconds())
                     / 3600)
        tz_mer = 15 * tz_offset    
    # once upon a time there was a latitude
    cdef double Lc = wog * (tz_mer - longt) / 15.0  # Local correction of time
    lat *= np.pi / 180  # in rad
    try:
        # where we used to raise an exception
        doys = datetime2doy(list(date))  # if date is datetime
    except (TypeError, AttributeError, ValueError, NotImplementedError,
            IndexError):
        try:
            # ...or two
            # if date is string
            doys = np.array(
                datetime2doy(list(vg.times.str2datetime(date, in_format))))
        except (TypeError, IndexError):
            # those were the doys my friend, i thought they never end
            doys = date  # if date is already in doys
    cdef double[:] hours = (doys - doys.astype(int)) * 24
    cdef long[:] doys_int = doys.astype(int)
    cdef double sinlat = sin(lat)
    cdef double coslat = cos(lat)
    for i in prange(n, nogil=True, schedule="static"):
        j1 = doys_int[i] / 100.
        j2 = (doys_int[i] - 180) / 100.
        if doys_int[i] > 180:
            Et = (-0.05039 - 0.33954 * j2 + 0.04084 * j2 ** 2 + 1.8928
                  * j2 ** 3 - 1.7619 * j2 ** 4 + 0.4224 * j2 ** 5)
        else:
            Et = (-0.04056 - 0.74503 * j1 + 0.08823 * j1 ** 2 + 2.0516
                  * j1 ** 3 - 1.8111 * j1 ** 4 + 0.42832 * j1 ** 5)
        t0 = 12 - Lc - Et
        # we sang and dance forever and a doy (less)
        # la la la la lala lala lala lala
        la = 2 * pi / 365 * (doys_int[i] - 1)
        d = (0.006918 - 0.399912 * cos(la) + 0.070257 * sin(la) -
             0.006758 * cos(2 * la) + 0.000907 * sin(2 * la) -
             0.002697 * cos(3 * la) + 0.00148 * sin(3 * la))
        sinphi = (sin(d) * sinlat +
                  cos(d) * coslat *
                  cos(15 * pi / 180.0 * (hours[i] - t0)))
        if sinphi < 0:
            sinphi = 0
        smax[i] = s0 * sinphi
    return smax


cdef pot_s_rad_c(double[:] doys,
                 float lon,
                 float lat,
                 float tz_mer,
                 float wog):
    cdef double j1, j2, Et, t0, la, d, sinphi, hour
    cdef double sinla, cosla, sin2la, cos2la, cosla2, sin3la, cos3la
    cdef int i, doy_int
    cdef int n = len(doys)
    cdef double pi = np.pi
    cdef double[::] smax = np.empty_like(doys)
    cdef int s0 = 1373  # Solarkonstante W/m^2
    # once upon a time there was a latitude
    # Local correction of time
    cdef double Lc = wog * (tz_mer - lon) / 15.0 
    lat *= np.pi / 180  # in rad
    # cdef double[:] hours = (doys - doys.astype(int)) * 24
    # cdef long[:] doys_int = doys.astype(int)
    cdef double sinlat = sin(lat)
    cdef double coslat = cos(lat)
    cdef double twopi365 = 2 * pi / 365
    # cdef float[:] j2_facts = np.array([-0.05039, -0.33954, 0.04084,
    #                                     1.8928, -1.7619, 0.4224],
    #                                   dtype=np.float32)
    # cdef float[:] j1_facts = np.array([-0.04056, -0.74503, 0.08823,
    #                                     2.0516, -1.8111, 0.42832],
    #                                   dtype=np.float32)
    # for i in prange(n, nogil=True, schedule="static"):
    for i in xrange(n):
        doy_int = <int> doys[i]
        hour = (doys[i] - doy_int) * 24
        j1 = doy_int / 100.
        j2 = (doy_int - 180) / 100.
        # if doy_int > 180:
        #     Et = j2_facts[0] + j2_facts[1] * j2
        #     for i in xrange(2, 6):
        #         j2 *= j2
        #         Et += j2_facts[i] * j2
        # else:
        #     Et = j1_facts[0] + j1_facts[1] * j1
        #     for i in xrange(2, 6):
        #         j1 *= j1
        #         Et += j1_facts[i] * j1
        if doy_int > 180:
            Et = (-0.05039 - 0.33954 * j2 + 0.04084 * j2 ** 2 + 1.8928
                  * j2 ** 3 - 1.7619 * j2 ** 4 + 0.4224 * j2 ** 5)
        else:
            Et = (-0.04056 - 0.74503 * j1 + 0.08823 * j1 ** 2 + 2.0516
                  * j1 ** 3 - 1.8111 * j1 ** 4 + 0.42832 * j1 ** 5)
        t0 = 12 - Lc - Et
        # we sang and dance forever and a doy (less)
        # la la la la lala lala lala lala
        # la = 2 * pi / 365 * (doy_int - 1)
        la = twopi365 * (doy_int - 1)
        sinla = sin(la)
        # vdSin(<int> 1, &la, &sinla)
        cosla = cos(la)
        sin2la = 2 * sinla * cosla
        cosla2 = cosla ** 2
        cos2la = cosla2 - sinla ** 2
        sin3la = sinla * (4 * cosla2 - 1)
        cos3la = 4 * cosla * cosla2 - 3 * cosla
        d = (0.006918 - 0.399912 * cosla + 0.070257 * sinla -
             0.006758 * cos2la + 0.000907 * sin2la -
             0.002697 * cos3la + 0.00148 * sin3la)
        sinphi = (sin(d) * sinlat +
                  cos(d) * coslat *
                  cos(15 * pi / 180.0 * (hour - t0)))
        if sinphi < 0:
            smax[i] = 0
        else:
            smax[i] = s0 * sinphi
    return smax


def sunshine_pot(doys, lat, longt, tz_mer=15.0, wog=-1):
    """Maximum daily sunshine hours based on evaluating pot_s_rad per minute.
    """
    cdef int mins_per_day = 24 * 60
    # doys = doys.astype(float)
    doys = doys[doys == doys.astype(int)]
    doys_minutes = doys.repeat(mins_per_day).reshape(-1, mins_per_day)
    doys_minutes += (np.arange(mins_per_day) /
                     float(mins_per_day))[None, :]
    doys_minutes = doys_minutes.ravel()
    smax = pot_s_rad_c(doys_minutes, longt, lat, tz_mer, wog)
    smax = np.array(smax).reshape(-1, mins_per_day)
    sun_hours = np.sum(smax > 0, axis=1) / 60.
    return sun_hours


cdef inline double mean(np.uint8_t[:] array,
                        unsigned int n) nogil:
    cdef unsigned int i
    cdef double sum = array[0]
    for i in range(1, n):
        sum += array[i]
    return sum / n

# cdef inline brunner_compound_(double[:] Ta_ranks,
#                               double[:] P_ranks,
#                               int n,
#                               double[:] bc):
cdef inline brunner_compound_(np.ndarray[DTYPE_t, ndim=1] Ta_ranks,
                              np.ndarray[DTYPE_t, ndim=1] P_ranks,
                              int n,
                              np.ndarray[DTYPE_t, ndim=1] bc):
    """Rank-based hot-dry index.


    Notes
    -----
    Brunner 2021 uses E-GPD for precipitation and a STI index for
    temperature. This implementation just uses empirical ranks.

    References
    ----------
    Brunner, Manuela I., Eric Gilleland, and Andrew W. Wood.
    “Space–Time Dependence of Compound Hot–Dry Events in the United
    States: Assessment Using a Multi-Site Multi-Variable Weather
    Generator.” Earth System Dynamics 12, no. 2 (May 19, 2021):
    621–34. https://doi.org/10.5194/esd-12-621-2021.
    """
    cdef unsigned int i, j
    cdef double Ta_rank, P_rank
    with nogil:
        for i in range(n):
            Ta_rank = Ta_ranks[i]
            P_rank = P_ranks[i]
            # bc[i] = mean((Ta_ranks <= Ta_rank) & (P_ranks <= P_rank),
            #              n)
            for j in range(n):
                # if (Ta_ranks[j] <= Ta_ranks[i]) and (P_ranks[j] <= P_ranks[i]):
                if (Ta_ranks[j] <= Ta_rank) and (P_ranks[j] <= P_rank):
                    bc[i] += 1
            bc[i] /= n

def brunner_compound(Ta, P, progress=False):
    if progress:
        from tqdm import tqdm as progress
    else:
        def progress(x, *args, **kwds):
            return x
    cdef unsigned int row_i
    Ta_ranks = bottleneck.rankdata(Ta, axis=-1)
    P_ranks = bottleneck.rankdata(np.negative(P), axis=-1)
    shape_before = Ta_ranks.shape
    cdef int T = shape_before[len(shape_before) - 1]
    Ta_ranks = Ta_ranks.reshape(-1, T)
    P_ranks = P_ranks.reshape(-1, T)
    cdef int n_rows = len(Ta_ranks)
    bc = np.zeros_like(Ta_ranks)
    for row_i in progress(range(n_rows), total=n_rows):
        brunner_compound_(Ta_ranks[row_i], P_ranks[row_i], T, bc[row_i])
    bc = bc.reshape(shape_before)
    return bc

