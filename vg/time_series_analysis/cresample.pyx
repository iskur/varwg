# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os
import shelve
import sys
from contextlib import closing

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.pyplot as plt
import cython
from vg import times, ctimes
from vg import helpers as my
from cython.parallel import prange, parallel


PY2 = sys.version_info.major == 2
shelve_filename = ("resampler_cache_{version}.sh"
                   .format(version="p2" if PY2 else "py3"))


def bias2mean(bias, a, b, c, d):
    return a * (1 - np.exp(-bias ** b / c)) + d


def mean2bias(mean, a, b, c, d):
    mean = np.atleast_1d(mean)
    bias = np.empty_like(mean)
    too_big_mask = mean > (a + d)
    bias[too_big_mask] = (-c * np.log(1e-9)) ** (1 / b)
    too_small_mask = mean < d
    bias[too_small_mask] = 0
    valid_mask = ~too_big_mask & ~too_small_mask
    bias[valid_mask] = ((-c * np.log(1 - (mean[valid_mask] - d) / a))
                        ** (1 / b))
    return bias


def _calibrate(**res_kwds):
    del res_kwds["n_sim_steps"]
    data = res_kwds["data"]
    theta_i = res_kwds["theta_i"]
    try:
        theta_i = res_kwds["theta_i"][0]
    except TypeError:
        pass
    theta = data[theta_i]
    # in order to have a roughly equally spaced mean array
    # means_incrs = np.linspace(0.5, 5, 12, endpoint=False)
    # theta_incrs = (-16.583 * np.log(1 - means_incrs /
    #                                 (1.1 * max(means_incrs)))) ** (1 / .513)
    # theta_incrs = np.concatenate((np.linspace(0, 1, 10),
    #                               np.linspace(2, 8, 5)))

    # theta_incr = res_kwds.pop("theta_incr")
    # theta_min = theta_incr.min()
    # biases = theta_incr - theta_min + np.linspace(0, 5, 15)[:, None]

    theta_incrs = np.concatenate((np.linspace(0, 1, 10),
                                  np.linspace(1.1, 8, 5)))
    biases = theta_incrs
    # theta_incrs = np.linspace(0, 4, 15)

    means = []
    res_kwds["recalibrate"] = False
    for bias in biases:
        res_kwds["bias"] = bias
        # res_kwds["theta_incr"] = bias
        res, _ = resample(**res_kwds)
        theta_res = res[theta_i]
        means += [np.mean(theta_res)]
    mean_diffs = np.array(means) - np.mean(theta)
    # biases = biases.mean(axis=1)

    def error(args):
        a, b, c, d = args
        mean_diffs_est = bias2mean(biases, a, b, c, d)
        return np.sum((mean_diffs - mean_diffs_est) ** 2)

    result = optimize.minimize(error,
                               (max(mean_diffs) + min(mean_diffs),
                                .5, 1., min(mean_diffs)),
                               # options=dict(disp=True)
                               )

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(biases, mean_diffs)
    # a, b, c, d = result.x
    # ax.plot(biases, bias2mean(biases, a, b, c, d))
    # ax.set_xlabel("biases")
    # ax.set_ylabel("mean diff")
    # ax.set_title("a: %.3f, b: %.3f, c: %.3f, d: %.3f" % (a, b, c, d))
    # plt.show()

    return result.x


def _transform_theta_incr(theta_incr, cache_dir=None, verbose=False,
                          recalibrate=False, **res_kwds):
    """Calibrate theta_incr in order to get a specific mean increase in
    temperature.

    Fits a curve of the following form:
      x = (- d \ln |1 - a / b|)^{1/c}
    """
    if cache_dir:
        shelve_filepath = os.path.join(cache_dir, shelve_filename)
    else:
        shelve_filepath = shelve_filename
    with closing(shelve.open(shelve_filepath, "c")) as sh:
        par_key = "{p}_{n_candidates}_{doy_tolerance}".format(**res_kwds)
        par_key_pos = par_key + "pos"
        if not recalibrate and par_key_pos in sh:
            a_pos, b_pos, c_pos, d_pos = sh[par_key_pos]
        else:
            if verbose:
                print("Calibrating resampler bias (positive leg)!")
            res_kwds["theta_incr"] = theta_incr
            a_pos, b_pos, c_pos, d_pos = _calibrate(**res_kwds)
            sh[par_key_pos] = a_pos, b_pos, c_pos, d_pos

        par_key_neg = par_key + "neg"
        if not recalibrate and par_key_neg in sh:
            a_neg, b_neg, c_neg, d_neg = sh[par_key_neg]
        else:
            if verbose:
                print("Calibrating resampler bias (negative leg)!")
            # calibration for negative mean changes
            res_kwds["theta_incr"] = theta_incr
            a_neg, b_neg, c_neg, d_neg = _calibrate(
                **{key: -val if key == "data" else val
                   for key, val in list(res_kwds.items())})
            sh[par_key_neg] = a_neg, b_neg, c_neg, d_neg

    bias_pos = mean2bias(theta_incr, a_pos, b_pos, c_pos, d_pos)
    # print("re-transformed bias: %.3f" %
    #       bias2mean(bias_pos, a_pos, b_pos, c_pos, d_pos))
    bias_neg = mean2bias(-theta_incr, a_neg, b_neg, c_neg, d_neg)
    bias = np.where(theta_incr > 0, bias_pos, -bias_neg)

    # if verbose:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(theta_incr[0], "-x")
    #     plt.plot(bias[0], "-x")
    #     plt.figure()
    #     th = np.linspace(-8, 8, 500)
    #     _bias_pos = calc_d(th, a_pos, b_pos, c_pos, d_pos)
    #     _bias_neg = calc_d(-th, a_neg, b_neg, c_neg, d_neg)
    #     _bias = np.where(th >= d_pos, _bias_pos, -_bias_neg)
    #     plt.plot(th, _bias)
    #     plt.show()

    # # do not fiddle with the data, where no change is requested
    # return np.where(np.isclose(theta_incr, 0), 0, bias)
    # do fiddle with the data, resampler can have inherent bias
    return bias


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double[::1] get_diffs(double[:, :, :] data_chunked,
                                  double[:, :] now):
    cdef unsigned int i, j, t
    cdef double diff_single
    K = data_chunked.shape[0]
    p = data_chunked.shape[1]
    n_chunks = data_chunked.shape[2]
    diffs = np.zeros(n_chunks, dtype=np.double)
    cdef double[::1] diffs_view = diffs
    for i in prange(K, nogil=True, schedule="guided"):
        for t in range(p):
            for j in range(n_chunks):
                diff_single = data_chunked[i, t, j] - now[i, t]
                diffs_view[j] += diff_single * diff_single
    return diffs


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef inline get_candidates(double[:, :, :] data_chunked,
#                            double[:, :] now,
#                            int n_candidates):
#     cdef unsigned int i, j, t, max_i
#     cdef long[::1] candidate_indices
#     cdef double diff, max_current = np.inf
#     cdef double[::1] diffs, candidates
#     K = data_chunked.shape[0]
#     p = data_chunked.shape[1]
#     n_chunks = data_chunked.shape[2]
#     diffs = np.full(n_candidates, np.inf, dtype=np.double)
#     cdef double[::1] diffs_view = diffs
#     candidate_indices = np.arange(n_candidates)
#     cdef long[::1] candidate_indices_view = candidate_indices
#     for j in range(n_chunks):
#         diff = 0
#         for i in range(K):
#             for t in range(p):
#                 diff += (data_chunked[i, t, j] - now[i, t]) ** 2
#                 if diff > max_current:
#                     continue
#             if diff > max_current:
#                 continue
#         if diff < max_current:
#             diffs_view[max_i] = diff
#             candidate_indices_view[max_i] = j
#             max_i = np.argmax(diffs)
#             max_current = diffs_view[max_i]
#     return np.array(candidate_indices)[np.argsort(diffs)]


cpdef resample(data, dtimes, p=3, n_sim_steps=None, theta_incr=0.0,
               theta_i=0, n_candidates=None, doy_tolerance=10,
               cache_dir=None, bias=None, verbose=False,
               return_candidates=False, recalibrate=False):
    """A simple multivariate resampler.
    Assumes that all variables are in the same dimension as similarity is
    evaluated in terms of summed absolute differences.

    Parameters
    ----------
    data : (K, T) float ndarray
        input data to mimic
        K - number of variables, T - number of time steps
    dtimes : (T,) ndarray of datetime objects
        dates of data.
    p : int, optional
        number of previous time steps to consider when searching for candidates
    n_sim_steps : int, optional
        number of time steps to simulate.
        If None, T is used (same as in `data`)
    theta_incr : scalar, (T,)array or None
        increase in average of temperature [sigma]
    theta_i : int
        row-index of temperature in data.  needed if theta_incr is not None.
    n_candidates : int or None, optional
        number of candidates to consider when choosing a similar time step.
        If None, it will be chosen as the square root of `T`.
    cache_dir : str or None, optional
        directory where calibration result is stored.
    calibrate : bias, optional
        Transformed theta_incr (used during calibration)
    return_candidates : bool, optional
        return the resampling candidates for every time step.

    Returns
    -------
    sim : (K, n_sim_steps) float ndarray
        Resampled time series.
    chosen_indices : (n_sim_steps,) int ndarray
        Resampled indices from `data`
    candidate_series : (K, n_sim_steps, n_candidates) double ndarray
        Candidates at every time step.
    """
    cdef int t
    if recalibrate or bias is None:
        cal_kwds = dict(my.ADict(locals()) - ("return_candidates", "t"))
        bias = _transform_theta_incr(**cal_kwds)
        if verbose:
            print("Transformed theta_incr=%.3f to bias=%.5f"
                  % (np.mean(theta_incr), np.mean(bias)))
    if n_sim_steps is None:
        n_sim_steps = data.shape[1]
    K, T = data.shape
    if n_candidates is None:
        n_candidates = int(np.sqrt(T))
        if verbose:
            print(f"Setting n_candidates={n_candidates}")
    candidate_series = np.empty((K, n_sim_steps, n_candidates))
    chosen_indices = np.empty(n_sim_steps, dtype=int)
    doys = times.datetime2doy(dtimes)
    # this is the hardest part
    # want to have T-1 chunks of (K, p) - shape
    # time domain indices, roughly in the form:
    # [(0,..,p), (1,...,p+1), ... (T-2*p,...,T-p)], but transposed!
    ii = np.arange(p)[:, None] + np.arange(data.shape[1] - p)[None, :]
    # this is now (K, p, T-p) - shape
    data_chunked = data[:, ii]

    bias_arr = np.zeros((K, n_sim_steps))
    bias_arr[theta_i, :] = np.atleast_2d(bias)
    ii = np.arange(p)[:, None] + np.arange(n_sim_steps - p)[None, :]
    bias_chunked = bias_arr[:, ii]

    sim = np.empty((data.shape[0], n_sim_steps))
    # start with something present in the data
    doy_neighbors = np.where(ctimes.doy_distance(doys[0], doys[:-p]) <=
                             doy_tolerance)[0]
    start_i = np.random.choice(doy_neighbors)
    sim[:, :p] = data[:, start_i:start_i + p]
    candidate_series[:, :p] = sim[:, :p, None]
    chosen_indices[:p] = list(range(start_i, start_i + p))

    # probability for choosing candidates, giving close candidates a
    # higher chance
    # see
    # Lall, U., & Sharma, A. (1996). A Nearest Neighbor Bootstrap For
    # Resampling Hydrologic Time Series. Water Resources Research,
    # 32(3), 679–693. http://doi.org/10.1029/95WR02966
    k_ji = 1. / np.arange(1, n_candidates + 1)
    k_ji /= np.sum(k_ji)

    for t in range(p, n_sim_steps):
        now = sim[:, t - p:t]
        # find time steps not far in terms of doy
        doy_distance = ctimes.doy_distance(doys[t % len(doys)], doys[:-p])
        doy_neighbors = np.where(doy_distance <= doy_tolerance)[0]
        # which bias do we use here?
        if theta_incr is not None:
            bias_cur = bias_chunked[..., t - p]
        else:
            bias_cur = 0
        # what looks most like now?
        diffs = get_diffs(data_chunked[..., doy_neighbors], now + bias_cur)
        candidates_i = np.argpartition(diffs, n_candidates)[:n_candidates]
        # candidates_i = candidates_i[np.argsort(diffs[candidates_i])]
        sort_i = np.argsort(np.array(diffs)[candidates_i])
        candidates_i = candidates_i[sort_i]
        if return_candidates:
            neighbors = doy_neighbors[candidates_i]
            candidate_series[:, t, :] = data[:, neighbors + p]
        # candidates = get_candidates(data_chunked[..., doy_neighbors],
        #                             now - bias, n_candidates)
        doy_neighbor_i = np.random.choice(candidates_i, p=k_ji)
        neighbor_i = doy_neighbors[doy_neighbor_i]
        # save the origin for outside analysis
        chosen_indices[t] = neighbor_i
        # i points at the beginning of a chunk, we want the values one time
        # step further
        sim[:, t] = data[:, neighbor_i + p]

    if return_candidates:
        return sim, chosen_indices, candidate_series
    return sim, chosen_indices


if __name__ == "__main__":
    import vg
    vg.conf = vg.vg_base.conf = vg.config_konstanz_disag
    met_vg = vg.VG((
        # "R",
        "theta", "Qsw",
        "ILWR", "rh", "u", "v"
                    ),
                   verbose=False)
    met_vg.fit(p=3)
    simt, sim = met_vg.simulate(res_kwds=dict(recalibrate=False,
                                              cy=False,
                                              n_candidates=20),
                                theta_incr=0.)

    theta_incrs = np.linspace(-10, 10, 21)
    data_mean = np.mean(met_vg.data_raw[0]) / 24.
    delta_means = []
    for theta_incr in theta_incrs:
        simt, sim = met_vg.simulate(res_kwds=dict(cy=False,
                                                  n_candidates=20),
                                    theta_incr=theta_incr)
        delta_means += [np.mean(sim[0]) - data_mean]
        print(theta_incr, delta_means[-1])

    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    ax.plot(theta_incrs, delta_means, "-x")
    ax.plot(theta_incrs, theta_incrs, color="gray")
    plt.show()
    
    # simt, sim = met_vg.simulate(T=365, resample=True, theta_incr=2)
    # met_vg.plot_autocorr()
    # from vg.time_series_analysis import time_series as ts
    # ts.plot_auto_corr([data, res], 14, var_names=met_vg.var_names)
    # ts.plt.show()
