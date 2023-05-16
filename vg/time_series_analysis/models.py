"""Various functions for time series anlysis.
If not specified differently, references are given on Luetkepohl "New
Introduction to Multiple Time Series Analysis".

.. currentmodule:: vg.time_series_analysis.models

.. autosummary::
   :nosignatures:
   :toctree: generated/

   VAR_LS
   VAR_LS_sim
   VAREX_LS
   VAREX_LS_sim
   VAR_order_selection
   VAR_residuals
   VAREX_residuals
   VAR_LS_predict
   AIC
   FPE
   HQ
   SC
"""
from __future__ import division, print_function

import itertools
from builtins import range, str, zip
from collections import namedtuple

import numpy as np
import scipy
from past.utils import old_div
from scipy import linalg, optimize
from scipy.linalg import kron
from scipy.stats import (skew, rankdata,
                         distributions as sp_distributions)
from tqdm import tqdm

from vg import helpers as my
from vg.time_series_analysis import time_series as ts
from vg.time_series_analysis.distributions import MDFt, norm
from vg.time_series_analysis import phase_randomization


mgarch_param_factory = namedtuple("mgarch_params",
                                  ("gamma0", "Gammas", "Gs", "cov_residuals"))
rng = np.random.default_rng()


def MGARCH_ML(residuals, q, m):
    """Estimating MGARCH parameters by numerically maximizing the
    log-likelihood.

    """
    K, T = residuals.shape
    dim0 = int(.5 * K * (K + 1))
    gamma0 = np.full(dim0, 1e-6)
    Gammas = q * [np.diagflat(np.full(dim0, .1))]
    Gs = m * [np.diagflat(np.full(dim0, .1))]
    delta_vec = vec(np.hstack((gamma0[:, None],
                               np.hstack(Gammas),
                               np.hstack(Gs))))
    sigma0 = np.cov(residuals)
    result = optimize.minimize(_MGARCH_likelihood,
                               x0=delta_vec,
                               bounds=len(delta_vec) * [(0, None)],
                               args=(residuals, q, m, sigma0),
                               options=dict(disp=True))
    gamma0, Gammas, Gs = _MGARCH_unpack(result.x, K, q, m)
    _check_stationarity(Gammas, Gs)
    mgarch_residuals = MGARCH_residuals(residuals, gamma0, Gammas, Gs)
    cov_residuals = np.cov(mgarch_residuals)
    params = mgarch_param_factory(gamma0, Gammas, Gs, cov_residuals)
    return params


def _check_stationarity(Gammas, Gs):
    M = np.zeros_like(Gammas[0])
    for array in itertools.chain(Gammas, Gs):
        M += array
    eigenvalues = linalg.eigvals(M)
    if np.any(eigenvalues >= 1):
        msg = "Non-stationarity: MGARCH eigenvalues too high. "
        msg = msg + "\n" + repr(eigenvalues)
        raise ValueError(msg)


def _MGARCH_likelihood(params, ut, q, m, sigma0):
    K, T = ut.shape
    gamma0, Gammas, Gs = _MGARCH_unpack(params, K, q, m)
    sigmas = m * [sigma0]
    llh = -K * np.log(2 * np.pi) * T

    def llh_single(t):
        llh = 0
        sigma_vech = gamma0
        for j in range(q):
            sigma_vech += Gammas[j] * vech(ut[:, t - j] * ut[:, t - j].T)
        for j in range(m):
            sigma_vech += Gs[j] * vech(sigmas[m - j - 1])
        sigma = unvech(sigma_vech, K)
        sigmas[:-1] = sigmas[1:]
        sigmas[-1] = sigma

        # punishment for mathematically unsound behaviour
        variance_min = np.diagonal(sigma).min()
        try:
            sigma_det = linalg.det(sigma)
            # print(sigma_det)
        except ValueError:
            sigma_det = -np.inf

        if variance_min <= 0:
            raise OverflowError
            # import ipdb; ipdb.set_trace()
            llh += variance_min ** 2
        elif sigma_det <= 0:
            # import ipdb; ipdb.set_trace()
            llh += sigma_det ** 2
        elif not np.isfinite(sigma_det):
            # import ipdb; ipdb.set_trace()
            llh += 1e9
        else:
            llh += (-np.log(sigma_det)
                    - np.dot(np.dot(ut[:, t].T, linalg.inv(sigma)),
                             ut[:, t]))
        return llh

    for t in range(max(q, m), T):
        try:
            llh_t = llh_single(t)
        except OverflowError:
            # this likelihood is bad enough, no need to go on...
            break
        llh += llh_t

    # llh *= .5
    return -llh


def MGARCH_residuals(ut, gamma0, Gammas, Gs):
    K, T = ut.shape
    residuals = np.zeros((K, T), dtype=float)
    q, m = len(Gammas), len(Gs)
    sigmas = m * [np.cov(ut)]
    for t in range(max(q, m), T):
        sigma_vech = gamma0
        for j in range(q):
            sigma_vech += Gammas[j] * vech(ut[:, t - j] * ut[:, t - j].T)
        for j in range(m):
            sigma_vech += Gs[j] * vech(sigmas[m - j - 1])
        sigma = unvech(sigma_vech, K)
        sigmas[:-1] = sigmas[1:]
        sigmas[-1] = sigma
        # residuals[:, t] = np.squeeze(sigma.dot(sigma) * ut[:, t])
        residuals[:, t] = np.squeeze(linalg.inv(sqrtm(sigma)) * ut[:, t])
    return residuals


def sqrtm(A):
    """Square root of a positive-definite Matrix.

    >>> A = np.array([[4., 0, 0], [0, 9., 0], [0, 0, 16.]])
    >>> sqrtm(A)
    array([[2., 0., 0.],
           [0., 3., 0.],
           [0., 0., 4.]])

    Notes
    -----
    See Appendix 9.4.
    """
    eigenvalues, P = linalg.eigh(A)
    Lambda_sqrt = np.diag(np.sqrt(eigenvalues))
    return P @ Lambda_sqrt @ P.T


def MGARCH_sim(params, T, sigma0, epsilon=None, n_presim_steps=100):
    gamma0, Gammas, Gs, cov_residuals = params
    q, m = len(Gammas), len(Gs)
    K = cov_residuals.shape[0]
    n_sim_steps = T + max(q, m) + n_presim_steps
    if epsilon is None:
        epsilon = np.random.multivariate_normal(K * [0], cov_residuals,
                                                n_sim_steps - max(q, m))
        epsilon = np.asmatrix(epsilon.T)
    ut = np.zeros((K, n_sim_steps))
    ut[:, :max(q, m)] = epsilon[:, :max(q, m)]
    sigmas = q * [sigma0]
    for t in range(max(q, m), n_sim_steps):
        sigma_vech = gamma0
        for j in range(q):
            sigma_vech += Gammas[j] @ vech(ut[:, t - j] @ ut[:, t - j].T)
        for j in range(m):
            sigma_vech += Gs[j] @ vech(sigmas[m - j - 1])
        sigma = unvech(sigma_vech, K)
        sigmas[:-1] = sigmas[1:]
        sigmas[-1] = sigma
        ut[:, t] = linalg.sqrtm(sigma) @ epsilon[t]
    return ut[:, -T:]


def _MGARCH_unpack(params, K, q, m):
    dim0 = int(.5 * K * (K + 1))
    delta = unvec(params, dim0)
    gamma0 = delta[:, 0]
    Gammas = [delta[:, i:i + dim0]
              for i in range(1, 1 + q * dim0, dim0)]
    Gs = [delta[:, i:i + dim0]
          for i in range(1 + q * dim0, 1 + (q + m) * dim0, dim0)]
    return gamma0, Gammas, Gs


def VAR_LS(data, p=2):
    """Least-Squares parameter estimation for a vector auto-regressive model of
    the form Y = B*Z + U. Records containing nans are excluded.
    Refer to the Least-Squares Estimator example 3.2.3 p.78. for method and
    variable names.

    Parameters
    ----------
    data : (K, T) ndarray
        K is the number of variables, T the number of timesteps
    p : int
        Autoregressive order of the process.

    Returns
    -------
    B : array
        Parameters of the fitted VAR-process.
    sigma_u: array
        Covariance matrix of the residuals.

    See also
    --------
    VAR_order_selection : Helps to find a p for parsimonious estimation.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_sim : Simulation based on LS estimator.
    VAR_LS_predict : Predict given prior data and LS estimator.
    """
    # number of variables
    if np.ndim(data) == 2:
        K, T = data.shape[0], data.shape[1] - p
    elif np.ndim(data) == 1:
        K, T = 1, len(data) - p
        data = data[np.newaxis, :]
    # Y is a (K, T - p) array
    Y = data[:, p:]

    Z = np.empty((K * p + 1, T))
    Zt = np.empty(K * p + 1)
    Zt[0] = 1
    for t in range(p, T + p):
        for subt in range(p):
            start_i = 1 + subt * K
            stop_i = 1 + (subt + 1) * K
            Zt[start_i:stop_i] = data[:, t - subt - 1]
        Z[:, t - p] = Zt

    # delete all columns containing nans
    Y_nan_cols = np.where(np.isnan(Y))[1]
    Y = np.delete(Y, Y_nan_cols, axis=1)
    Z = np.delete(Z, Y_nan_cols, axis=1)
    Z_nan_cols = np.where(np.isnan(Z))[1]
    Y = np.delete(Y, Z_nan_cols, axis=1)
    Z = np.delete(Z, Z_nan_cols, axis=1)
    # no idea why there are remaining columns with nans in Z!
    Z_nan_cols_ = np.where(np.isnan(Z))[1]
    Y = np.delete(Y, Z_nan_cols_, axis=1)
    Z = np.delete(Z, Z_nan_cols_, axis=1)

    # B contains all the parameters we want: (nu, A1, ..., Ap)
    # Y = BZ + U
    try:
        inv = np.linalg.inv(Z @ Z.T)
    except np.linalg.LinAlgError:
        print("Adding a little random noise in order to invert matrix...")
        Z += (rng.normal(Z.size) * Z.std() * 1e-9).reshape(Z.shape)
        inv = np.linalg.inv(Z @ Z.T)
    B = Y @ Z.T @ inv

    # covariance matrix of the noise
    sigma_u = Y @ Y.T - B @ Z @ Y.T
    sigma_u /= (Y.shape[1] - K * p - 1)

    return B, sigma_u


def VAREX_LS(data, p, ex):
    """Least-Squares parameter estimation for a vector auto-regressive model of
    the form

    ..math::y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + C x_t + u_t

    Records containing nans are excluded.
    Refer to the Least-Squares Estimator example 3.2.3 p.78. for method and
    variable names.

    Parameters
    ----------
    data : (K, T) ndarray
        K is the number of variables, T the number of timesteps
    ex : (T,) ndarray
        An external variable
    p : int
        Autoregressive order of the process.

    Returns
    -------
    B : array
        Parameters of the fitted VAR-process.
        B := (A_1, ..., A_p, C)
    sigma_u: array
        Covariance matrix of the residuals of the data.

    See also
    --------
    VAR_order_selection : Helps to find a p for parsimonious estimation.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_sim : Simulation based on LS estimator.
    VAR_LS_predict : Predict given prior data and LS estimator.
    """
    # number of variables
    if np.ndim(data) == 2:
        K, T = data.shape[0], data.shape[1] - p
    elif np.ndim(data) == 1:
        K, T = 1, len(data) - p
        data = data[np.newaxis, :]
    # Y is a (K, T) array
    Y = data[:, p:]

    Z = np.empty((K * p + 1, T))
    Zt = np.empty((K * p, 1))
    Z[-1] = ex[p:].reshape(1, T)
    for t in range(p, T + p):
        for subt in range(p):
            start_i = subt * K
            stop_i = (subt + 1) * K
            Zt[start_i:stop_i] = data[:, t - subt - 1].reshape((K, 1))
        Z[:-1, t - p] = Zt

    # delete all columns containing nans
    Y_nan_cols = np.where(np.isnan(Y))[1]
    Y = np.delete(Y, Y_nan_cols, axis=1)
    Z = np.delete(Z, Y_nan_cols, axis=1)
    Z_nan_cols = np.where(np.isnan(Z))[1]
    Y = np.delete(Y, Z_nan_cols, axis=1)
    Z = np.delete(Z, Z_nan_cols, axis=1)

    # B contains all the parameters we want: (A1, ..., Ap, C)
    # Y = BZ + U
    B = Y @ Z.T @ np.linalg.inv(Z @ Z.T)

    # covariance matrix of the noise of data
    sigma_u = Y @ Y.T - B @ Z @ Y.T
    sigma_u /= Y_nan_cols.size - K * p - 1

    return B, sigma_u


def SVAR_LS(data, doys, p=2, doy_width=10, fft_order=4, var_names=None,
            verbose=True):
    """Seasonal version of the least squares estimator."""
    K, T = data.shape
    Bs, sigma_us = [], []
    unique_doys = np.unique(doys)
    for doy in tqdm(unique_doys, disable=(not verbose)):
        mask = (doys > doy - doy_width) & (doys <= doy + doy_width)
        if (doy - doy_width) < 0:
            mask |= doys > (365. - doy_width + doy)
        if (doy + doy_width) > 365:
            mask |= doys < (doy + doy_width - 365.)
        B, sigma_u = VAR_LS(np.where(mask, data, np.nan), p=p)
        Bs += [B]
        sigma_us += [sigma_u]

    Bs, sigma_us = np.asarray(Bs), np.asarray(sigma_us)

    # import matplotlib.pyplot as plt
    # import itertools
    # Bs = np.asarray(Bs)
    # sigma_us = np.asarray(sigma_us)
    # for i, j in itertools.combinations(range(data.shape[0]), 2):
    #     plt.figure()
    #     plt.plot(Bs[:, i, j + 1], "b")
    #     plt.plot(np.fft.irfft(np.fft.fft(Bs[:, i, j + 1])[:fft_order + 1],
    #                           n_doys_unique),
    #              "b", label="%s_t-1 -> %s_t" % (var_names[j], var_names[i]))
    #     plt.plot(Bs[:, j, i + 1], "g")
    #     plt.plot(np.fft.irfft(np.fft.fft(Bs[:, j, i + 1])[:fft_order + 1],
    #                           n_doys_unique),
    #              "g", label="%s_t-1 -> %s_t" % (var_names[i], var_names[j]))
    #     plt.plot(sigma_us[:, i, j])
    #     plt.plot(np.fft.irfft(np.fft.fft(sigma_us[:, i, j])[:fft_order + 1],
    #                           n_doys_unique))
    #     plt.legend()
    #     plt.title("%s %s" % (var_names[i], var_names[j]))
    #     plt.grid()
    # for i in range(data.shape[0]):
    #     plt.figure()
    #     plt.plot(Bs[:, i, i + 1], "b")
    #     plt.plot(np.fft.irfft(np.fft.fft(Bs[:, i, i + 1])[:fft_order + 1],
    #                           n_doys_unique),
    #              "b")
    #     plt.grid()
    #     plt.title(var_names[i])

    def matr_fft(M, fft_order):
        return np.asarray(
            [[my.fourier_approx(M[:, k, j], fft_order)
              for j in range(M.shape[2])]
             for k in range(K)])

    # import matplotlib.pyplot as plt
    # plt.rcParams["font.size"] = 7
    # plt.rcParams["legend.fontsize"] = 6
    # Bs = np.asarray(Bs)
    # Bs_app = matr_fft(Bs)
    # columnwidth_pt = 307.28987
    # inches_per_pt = 1. / 72.27
    # fig_width = columnwidth_pt * inches_per_pt
    # plt.figure(figsize=(fig_width, .5 * fig_width))
    # plt.plot(Bs[:, 0, 1], label="from moving window")
    # plt.plot(Bs_app[0, 1], label="fft approximated")
    # plt.grid()
    # plt.xlabel("day of year")
    # plt.tight_layout()
    # plt.savefig("/home/dirk/Desktop/Diss/Praesentationen/A00_approx",
    #             transparent=True, dpi=600)
    # plt.show()

    Bs = matr_fft(Bs, fft_order)
    sigma_us = matr_fft(sigma_us, fft_order)
    return Bs, sigma_us


def VAR_mean(B):
    K = B.shape[0]
    p = (B.shape[1] - 1) // K
    Ai = [np.asarray(B[:, 1 + i * K: 1 + (i + 1) * K])
          for i in range(p)]
    nu = B[:, 0]
    mu = np.identity(K)
    for i in range(p):
        mu -= Ai[i]
    return (np.linalg.inv(mu) @ nu.T)[:, None]


def VAR_cov(B, sigma_u):
    K = B.shape[0]
    p = (B.shape[1] - 1) // K
    A = B2A(B)
    Ikp2 = np.identity((K * p) ** 2)
    sigma_U = np.zeros_like(A)
    sigma_U[:K, :K] = sigma_u
    cov_vec = np.linalg.inv(Ikp2 - np.kron(A, A)) @ vec(sigma_U)
    return unvec(cov_vec, K)


def B2A(B):
    K = B.shape[0]
    p = (B.shape[1] - 1) // K
    A = np.zeros((K * p, K * p))
    A[:K] = B[:, 1:]
    Ik = np.identity(K)
    for i in range(p - 1):
        A[(i + 1) * K: (i + 2) * K,
          i * K: (i + 1) * K] = Ik
    return A


@my.cache("ut")
def SVAR_LS_sim(Bs, sigma_us, doys, m=None, ia=None, m_trend=None, u=None,
              n_presim_steps=100, fixed_data=None, phase_randomize=False,
              rphases=None, return_rphases=False, p_kwds=None,
              taboo_period_min=None, taboo_period_max=None):
    if p_kwds is None:
        p_kwds = dict()
    doys_ii = (doys % 365) / 365. * len(np.unique(doys))
    # doys_ii = doys % len(np.unique(doys))
    doys_ii = doys_ii.astype(int)
    K = Bs.shape[0]
    p = (Bs.shape[1] - 1) // K
    T = len(doys)
    Y = np.zeros((K, T + p))
    Y[:, :p] = VAR_mean(Bs[..., doys_ii[-1]])
    if phase_randomize:
        if u is None:
            raise RuntimeError("u must be passed for phase randomization!")
        u = phase_randomization.randomize2d(
            u,
            T=T,
            taboo_period_min=taboo_period_min,
            taboo_period_max=taboo_period_max,
            return_rphases=return_rphases,
            rphases=rphases,
            **p_kwds)
        if return_rphases:
            u, rphases = u
    if u is None:
        u = np.array(
            [rng.multivariate_normal(K * [0],
                                     sigma_us[..., doy_i])
             for doy_i in doys_ii])
        u = u.T
    SVAR_LS_sim.ut = u
    for t, doy_i in enumerate(doys_ii):
        # B = Bs[..., doy_i].copy()
        # if t > 0:
        #     for j in range(1, p):
        #         islice = slice(j * K + 1, (j + 1) * K + 1)
        #         # B[:, islice] = Bs[:, islice, doy_i - j]
        #         B[:, islice] = .5 * (B[:, islice] +
        #                              Bs[:, islice, doy_i - j])
        Y[:, t + p] = \
            VAR_LS_sim(
                Bs[..., doy_i],
                # ignored when using phase_randomization!
                sigma_us[..., doy_i],
                1,
                None if m is None else m[:, t, None],
                None if ia is None else ia[:, t, None],
                m_trend,
                n_presim_steps=0,
                # u=None if u is None else u[:, t, None],
                u=u[:, t, None],
                prev_data=Y[:, t:t + p],
                # u is already phase randomized above if requested
                phase_randomize=False,
            ).ravel()
    Y = Y[:, p:]
    Y_new = Y

    # # SVAR messes with the marginals, so we z-transform to normal as
    # # expected by the rest of the code base. We maintain the means,
    # # though.
    # means = np.array([np.squeeze(VAR_mean(Bs[..., doys_ii[t]]))
    #                   for t in range(T)]).T
    # stds = np.array([np.sqrt(np.diag(
    #     VAR_cov(Bs[..., doys_ii[t]],
    #             sigma_us[..., doys_ii[t]])))
    #          for t in range(T)]).T
    # means_mean = means.mean(axis=-1)[:, None]
    # stds_mean = stds.mean(axis=-1)[:, None]
    # Y_new = (Y - means_mean) / stds_mean * stds + means

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(nrows=K, ncols=2,
    #                         sharex=True, sharey="row",
    #                         constrained_layout=True)
    # for var_i in range(K):
    #     axs[var_i, 0].plot(Y[var_i])
    #     axs[var_i, 0].plot(means[var_i])
    #     axs[var_i, 0].plot(means[var_i] - stds[var_i], "--k")
    #     axs[var_i, 0].plot(means[var_i] + stds[var_i], "--k")
    #     axs[var_i, 0].axhline(means_mean[var_i])
    #     axs[var_i, 0].axhline(means_mean[var_i] - stds_mean[var_i],
    #                           linestyle="--", color="b")
    #     axs[var_i, 0].axhline(means_mean[var_i] + stds_mean[var_i],
    #                           linestyle="--", color="b")
    #     axs[var_i, 1].plot(Y_new[var_i], alpha=.25, label="new")
    #     axs[var_i, 1].plot(Y[var_i], alpha=.25, label="old")
    # axs[0, 1].legend(loc="best")
    # plt.show()

    if return_rphases:
        return Y_new, rphases
    return Y_new


def VAR_LS_asy(data, skewed_i, p=None):
    """Skewness of the differences of the residuals."""
    if p is None:
        p = VAR_order_selection(data)
    B, _ = VAR_LS(data, p)
    residuals = VAR_residuals(data, B, p)
    if isinstance(skewed_i, int):
        skewed_i = [skewed_i]
    asymmetries = []
    for i in skewed_i:
        diff = np.diff(residuals[i])
        asymmetries += [skew(diff[np.isfinite(diff)])]
    return asymmetries


@my.cache("ut")
def VAR_LS_sim_asy(B, sigma_u, T, data, p, skewed_i,
                   n_presim_steps=100, verbose=False, var_names=None,
                   *args, **kwds):
    residuals = VAR_residuals(data, B, p)
    data_skew = VAR_LS_asy(data, skewed_i, p)
    data_skew = np.squeeze(data_skew)
    K = data.shape[0]
    n_sim_steps = T + p + n_presim_steps
    # u = np.random.multivariate_normal(K * [0], sigma_u, n_sim_steps - p).T
    params = MDFt.fit(residuals)
    u = MDFt.sample(n_sim_steps, *params)
    u_initial = np.copy(u)

    VAR_LS_sim_asy.ut = u

    fig = my.splom(residuals)
    fig.suptitle("Residuals")
    fig = my.splom(u)
    fig.suptitle("Simulated residuals")

    # if True:
    #     return VAR_LS_sim(B, sigma_u, T, u=u, *args, **kwds)

    # u = np.asmatrix(u.T)
    if var_names is None:
        var_names = [str(i) for i in range(K)]

    def asy(u):
        return skew(np.diff(u[skewed_i], axis=1), axis=1)

    def skew_error(u):
        u_skew = asy(u)
        return np.sum((data_skew - u_skew) ** 2)

    @my.cache("i", "j", "k", "di", "dj", "dk")
    def swap(data):
        # i, j, k = np.random.randint(data.shape[1], size=3)
        width = 10
        j = rng.randint(1, data.shape[1] - 2)
        i = rng.randint(max(0, j - 2), j)
        k = rng.randint(j + 1, min(data.shape[1] - 1, j + width))
        di, dj, dk = data[skewed_i, [i, j, k]]
        if di < dj < dk:
            data[:, [j, k]] = dk, dj
        elif dk < di < dj:
            data[:, [i, k]] = dk, di
        elif dk < dj < di:
            data[:, [i, j, k]] = dk, di, dj
        elif dj < di < dk:
            data[:, [i, j, k]] = dj, dk, di
        swap.i, swap.j, swap.k = i, j, k
        swap.di, swap.dj, swap.dk = di, dj, dk
        return data

    def swap_back(data):
        data[:, [swap.i, swap.j, swap.k]] = swap.di, swap.dj, swap.dk
        return data

    # @my.cache("i", "j")
    # def swap(data):
    #     i, j = np.random.randint(data.shape[1], size=2)
    #     data[:, [i, j]] = data[:, [j, i]]
    #     swap.i, swap.j = i, j
    #     return data

    # def swap_back(data):
    #     data[:, [swap.j, swap.i]] = data[:, [swap.i, swap.j]]
    #     return data

    temp = 2.
    k = .95
    m, M = 0, 200
    error_old = skew_error(u)
    u = swap(u)
    error_new = skew_error(u)
    error_best = min(error_old, error_new)
    error_last_best = error_best
    u_best = np.copy(u)

    if verbose:
        print("Data asymmetry: %.3f" % data_skew)
        print("Initial sim asymmetry: %.3f" % asy(u))

    # while iteration < 100 or abs(error_new - error_old) / error_old > 1e-9:
    iteration = 0
    accept = np.empty(M, dtype=bool)
    while True:
        iteration += 1
        accept[m] = True
        if error_new > error_old:
            p = np.exp(old_div((error_old - error_new), temp))
            if not rng.rand() < p:
                accept[m] = False

        if error_new < error_best:
            error_last_best = error_best
            error_best = error_new
            u_best = np.copy(u)

        if accept[m]:
            error_old = error_new
            u = swap(u)
            error_new = skew_error(u)
        else:
            u = swap_back(u)

        m += 1
        if m == M:
            m = 0
            temp *= k
            print ("sim: %.3f, data: %.3f, temp: %.5f" %
                   (asy(u), data_skew, temp))
            if (old_div((error_last_best - error_best), error_last_best) < 1e-6
                    or (np.all(~accept) and iteration > 5000)):
                break

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    axs[0].plot(residuals[0], label="residuals")
    axs[0].plot(u_best[0], label="u")
    axs[0].legend(loc="best")
    axs[0].set_title("theta")

    axs[1].plot(residuals[1], label="residuals")
    axs[1].plot(u_best[1], label="u")
    axs[1].legend(loc="best")
    axs[1].set_title("U")

    axs[2].plot(residuals[2], label="residuals")
    axs[2].plot(u_best[2], label="u")
    axs[2].legend(loc="best")
    axs[2].set_title("wf")
    fig.suptitle("Residuals")

    for var_i in range(K):
        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        both = [
            residuals[var_i],
            # scipy.stats.trimboth(residuals[var_i], .05),
            np.squeeze(u_best[var_i])]
        labels = "measured", "simulated"
        for ax, vals, label in zip(axs, both, labels):
            my.hist(vals, 20, kde=True,
                    dist=(scipy.stats.distributions.norm,
                          scipy.stats.distributions.t),
                    ax=ax)
            skew_ = scipy.stats.skew(vals)
            kurt = scipy.stats.kurtosis(vals)
            ax.set_title(r"%s residuals $\gamma$=%.3f kurt=%.3f" %
                         (label, skew_, kurt))
        fig.suptitle(var_names[var_i])

    VAR_LS_sim_asy.ut = u_best
    return VAR_LS_sim(B, sigma_u, T, u=u_best, *args, **kwds)


def VAR_LS_sim(B, sigma_u, T, m=None, ia=None, m_trend=None, u=None,
               n_presim_steps=100, fixed_data=None, prev_data=None,
               transform=None, phase_randomize=False):
    """Based on a least squares estimator, simulate a time-series of the form
    ..math::y(t) = nu + A1*y(t-1) + ... + Ap*y(t-p) + ut
    B contains (nu, A1, ..., Ap).
    See p. 707f

    Parameters
    ----------
    B : (K, K*p+1) ndarray
        Parameters of the VAR-process as returned from VAR_LS. K is the number
        of variables, p the autoregressive order.
    sigma_u : (K, K) ndarray
        Covariance matrix of the residuals as returned from VAR_LS.
    T : int
        Number of timesteps to simulate.
    m : (K,) ndarray, optional
        Process means (will be scaled according to B).
    ia : (K, T) ndarray, optional
        Interannual variability. Additional time-varying disturbance to the
        process means (will be scaled according to B).
    m_trend : (K,) ndarray, optional
        Change in means, that will be applied linearly so that this change is
        reached after the T timesteps.
    u : (K, T) ndarray, optional
        Residuals to be used instead of multivariate gaussian serially
        independent random numbers.
    n_presim_steps : int, optional
        Number of presimulation timesteps that will be thrown away.
    fixed_data : (K, T) ndarray, optional
        Data that will be fixed, i.e. at each timestep, these will be put in
        instead of the actual simulated values. Where fixed_data is nan, the
        simulated values will not be overwritten.
    transform : callable, optional
        On-the-fly transformation that accepts a sequence of length K
        (values of all variables at one time step) and the time index t.

    Returns
    -------
    Y : (K, T) ndarray
        Simulated values.

    See also
    --------
    VAR_LS : Least-squares estimator (to get B and sigma_u).
    VAR_order_selection : Helps to find a p for parsimonious estimation.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_predict : Predict given prior data and LS estimator.
    """
    # number of variables
    K = B.shape[0]
    # order of VAR_LS-process
    p = (B.shape[1] - 1) // K
    n_sim_steps = T + p + n_presim_steps

    if m is not None:
        m = _scale_additive(m, B[:, 1:], p)
        # we have to expand m to include the pre-simulation timesteps
        m = np.concatenate((m[:, :n_presim_steps], m), axis=1)

    if ia is not None:
        ia = _scale_additive(ia, B[:, 1:], p)

    # the first p columns are initial values, which will be omitted later
    Y = np.zeros((K, n_sim_steps))
    if prev_data is not None:
        Y[:, :p] = prev_data[:, -p:]
        n_sim_steps -= n_presim_steps

    Ai = [np.asarray(B[:, 1 + i * K: 1 + (i + 1) * K]) for i in range(p)]

    if m is None and prev_data is None:
        # setting starting values to the process mean
        Y[:, :p] = VAR_mean(B)
    Y = np.asarray(Y)

    if phase_randomize:
        if u is None:
            raise RuntimeError("u must be passed for phase randomization!")
        u = phase_randomization.randomize2d(u, T=T,
                                            taboo_period_min=None,
                                            taboo_period_max=None)
    elif u is None:
        u = rng.multivariate_normal(K * [0], sigma_u, n_sim_steps - p)
        u = u.T

    Y[:, -u.shape[1]:] += u
    if m is None:
        nu = B[:, 0]
        Y[:, p:] += nu[:, None]
    elif m is not None:
        Y[:, -m.shape[1]:] += m

    if ia is not None:
        Y[:, -ia.shape[1]:] += ia

    if m_trend is not None:
        # apply changes as a trend
        m_trend = np.asarray(m_trend)[:, None]
        m_trend = _scale_additive(m_trend, B[:, 1:], p)
        Y[:, -T:] += np.arange(T, dtype=float) / T * m_trend

    start_t = n_sim_steps - T
    for t in range(p, n_sim_steps):
        for i in range(p):
            Y[:, t] += Ai[i] @ Y[:, t - i - 1]

        # on-line transformations
        if transform:
            Y[:, t] = transform(Y[:, t], t - start_t)

        if (fixed_data is not None) and (t >= start_t):
            # fixing what's asked to be held constant
            Y[:, t] = np.where(np.isnan(fixed_data[:, t - start_t]),
                               Y[:, t], fixed_data[:, t - start_t])
    return Y[:, -T:]


def VAREX_LS_sim(B, sigma_u, T, ex, m=None, ia=None, m_trend=None, u=None,
                 n_presim_steps=100, prev_data=None, ex_kwds=None):
    """Based on a least squares estimator, simulate a time-series of the form
    ..math::y(t) = A1*y(t-1) + ... + Ap*y(t-p) + C*x(t-1) + ut
    B contains (A1, ..., Ap, C).
    See p. 707f

    Parameters
    ----------
    B : (K, K*p+1) ndarray
        Parameters of the VAR-process as returned from VAR_LS. K is the number
        of variables, p the autoregressive order.
    sigma_u : (K, K) ndarray
        Covariance matrix of the residuals as returned from VAR_LS.
    ex : (T,) ndarray or function
        External variable. If given as a function, ex_t will be generated by
        calling ex(Y[:t], **ex_kwds), with Y being the simulated values.
    T : int
        Number of timesteps to simulate.
    m : (K,) ndarray, optional
        Process means (will be scaled according to B).
    ia : (K, T) ndarray, optional
        Interannual variability. Additional time-varying disturbance to the
        process means (will be scaled according to B).
    m_trend : (K,) ndarray, optional
        Change in means, that will be applied linearly so that this change is
        reached after the T timesteps.
    u : (K, T) ndarray, optional
        Residuals to be used instead of multivariate gaussian serially
        independent random numbers.
    n_presim_steps : int, optional
        Number of presimulation timesteps that will be thrown away.
    ex_kwds : dict, optional
        Keyword arguments to be passed to ex.

    Returns
    -------
    Y : (K, T) ndarray
        Simulated values.
    ex_out : (T,) ndarray
        External variable.

    See also
    --------
    VAR_LS : Least-squares estimator (to get B and sigma_u).
    VAR_order_selection : Helps to find a p for parsimonious estimation.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_predict : Predict given prior data and LS estimator.
    """
    # number of variables
    K = B.shape[0]
    # order of VAR_LS-process
    p = (B.shape[1] - 1) // K
    n_sim_steps = T + p + n_presim_steps

    try:
        len(ex)
        ex_isfunc = False
        ex_out = ex
    except TypeError:
        ex_kwds = {} if ex_kwds is None else ex_kwds
        ex_isfunc = True
        ex_out = np.empty(T)

    if m is not None:
        m = _scale_additive(m, B[:, 1:], p)
        # we have to expand m to include the pre-simulation timesteps
        m = np.concatenate((m[:, :n_presim_steps], m), axis=1)
    m_trend = np.asarray([0] * K) if m_trend is None else np.asarray(m_trend)
    m_trend = m_trend[:, np.newaxis]  # ha! erwischt!
    m_trend = _scale_additive(m_trend, B[:, 1:], p)

    if ia is not None:
        ia = _scale_additive(ia, B[:, 1:], p)

    # the first p columns are initial values, which will be omitted later
    Y = np.zeros((K, n_sim_steps))
    if prev_data is not None:
        Y[:, :p] = prev_data[:, -p:]
        n_sim_steps -= n_presim_steps

    Ai = [B[:, i * K: (i + 1) * K] for i in range(p)]
    C = B[:, -1]

    if u is None:
        u = rng.multivariate_normal(K * [0], sigma_u, n_sim_steps - p)
        u = u.T

    Y[:, -u.shape[1]:] += u
    if m is not None:
        Y[:, -m.shape[1]:] += m

    if ia is not None:
        Y[:, -ia.shape[1]:] += ia

    # apply changes as a trend
    Y[:, -T:] += np.arange(T, dtype=float) / T * m_trend

    start_t = n_sim_steps - T
    for t in range(p, n_sim_steps):
        for i in range(p):
            Y[:, t] += Ai[i] @ Y[:, t - i - 1]

        if t >= start_t:
            if ex_isfunc:
                ex_t = ex(Y[:, :t], **ex_kwds)
                ex_out[t - p - start_t] = ex_t
            else:
                ex_t = ex[t - p - start_t]
            Y[:, t] += np.squeeze(C * ex_t)

    return Y[:, -T:], ex_out


def VAR_residuals(data, B, p=2):
    K, T = data.shape

    # were we given a B with the nus (in the first column)?
    if B.shape[1] % (K * p) == 1:
        mean_adjusted = False
        i_shift = 1
        # what is the process mean?
        nu = B[:, 0]
        mu = VAR_mean(B)
        nu = np.asarray(nu).ravel()
    else:
        mean_adjusted = True
        i_shift = 0
        # estimate the process means from the data.
        mu = np.mean(data, axis=1)[:, None]

    # set the pre-sample period to the process means
    data = np.concatenate((np.empty((K, p)), data), axis=1)
    data[:, :p] = mu
    resi = np.copy(data)

    for t in range(p, T + p):
        for i in range(p):
            Ai = B[:, i_shift + i * K: i_shift + (i + 1) * K]
            resi[:, t] -= Ai @ data[:, t - i - 1]
        if not mean_adjusted:
            resi[:, t] -= nu
    return resi[:, p:] + mu if mean_adjusted else resi[:, p:]


def VAR_LS_extro(data, data_trans, transforms, backtransforms, p=2):
    B0, sigma_u = VAR_LS(data, p=p)
    B = np.copy(B0)
    K, T = data.shape
    M = B.shape[1]

    def errors(*Bs):
        B[:, 1:] = np.array(Bs).reshape((K, M - 1))
        pred = VAR_onestep_predictions(data, B, p=p)
        pred_trans = transforms(pred)
        return data_trans - pred_trans

    def error_sum(*Bs):
        return np.sum(errors(*Bs) ** 2)

    B_part = optimize.minimize(error_sum, x0=np.ravel(B0[:, 1:]),
                               options=dict(disp=True),
                               # method="Nelder-Mead",
                               method="Powell",
                               ).x
    B[:, 1:] = B_part.reshape((K, M - 1))
    residuals = VAR_LS_extro.residuals = errors(B[:, 1:])

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=K, sharex=True)
    pred = VAR_onestep_predictions(data, B, p=p)
    pred_trans = transforms(pred)
    for i, ax in enumerate(axs):
        ax.plot(data_trans[i], label="obs")
        ax.plot(pred_trans[i], label="est")
    axs[-1].legend()

    residuals_norm = backtransforms(residuals)
    finite_rows = np.all(np.isfinite(residuals_norm), axis=1)
    sigma_u = np.cov(residuals_norm[:, finite_rows])
    return B, sigma_u


def VAR_onestep_predictions(data, B, p=2):
    K, T = data.shape

    # were we given a B with the nus (in the first column)?
    if B.shape[1] % (K * p) == 1:
        mean_adjusted = False
        i_shift = 1
        # what is the process mean?
        nu = B[:, 0]
        mu = VAR_mean(B)
        nu = np.asarray(nu).ravel()
    else:
        mean_adjusted = True
        i_shift = 0
        # estimate the process means from the data.
        mu = np.mean(data, axis=1)[:, np.newaxis]

    predictions = np.zeros_like(data)
    predictions[:, :p] = mu
    # set the pre-sample period to the process means
    data = np.concatenate((np.empty((K, p)), data), axis=1)

    for t in range(p, T):
        for i in range(p):
            Ai = B[:, i_shift + i * K: i_shift + (i + 1) * K]
            predictions[:, t] += Ai @ data[:, t - i - 1]
        if not mean_adjusted:
            predictions[:, t] -= nu
    return (np.asarray(predictions + mu)
            if mean_adjusted
            else predictions)


def VAREX_residuals(data, ex, B, p=2, ex_kwds=None):
    K, T = data.shape

    try:
        len(ex)
        ex_isfunc = False
    except TypeError:
        ex_kwds = {} if ex_kwds is None else ex_kwds
        ex_isfunc = True

    # set the pre-sample period to the process means
    data = np.concatenate((np.zeros((K, p)), data), axis=1)
    resi = np.copy(data)

    C = np.asarray(B[:, -1])
    for t in range(p, T + p):
        for i in range(p):
            Ai = np.asarray(B[:, i * K: (i + 1) * K])
            resi[:, t] -= Ai @ data[:, t - i - 1]

        if ex_isfunc:
            ex_t = ex(data[:, :t], **ex_kwds)
        else:
            ex_t = ex[t - p]

        resi[:, t] -= np.squeeze(C * ex_t)
    return np.asarray(resi[:, p:])


def SVAR_residuals(data, doys, B, p=2):
    K, T = data.shape
    doys_ii = (doys % 365) / 365. * len(np.unique(doys))
    doys_ii = doys_ii.astype(int)

    # were we given a B with the nus (in the first column)?
    if B.shape[1] % (K * p) == 1:
        mean_adjusted = False
        i_shift = 1
    else:
        mean_adjusted = True
        i_shift = 0
        # estimate the process means from the data.
        mu = np.asmatrix(np.mean(data, axis=1)[:, np.newaxis])

    # set the pre-sample period to the process means
    data = np.concatenate((np.empty((K, p)), data), axis=1)
    data[:, :p] = VAR_mean(B[..., 0])
    resi = np.copy(data)

    for t in range(p, T + p):
        for i in range(p):
            Ai = B[:, i_shift + i * K: i_shift + (i + 1) * K, doys_ii[t - p]]
            resi[:, t] -= Ai @ data[:, t - i - 1]
    return resi[:, p:] + mu if mean_adjusted else resi[:, p:]


def VARMA_LS_prelim(data, p, q):
    """Preliminary version of the general simple least-squares vector
    autoregressive estimator for a vector autoregressive moving-average
    process. See p.474ff"""
    K, T = data.shape[0], data.shape[1] - p
    # number of parameters
    N = K ** 2 * (p + q)

    # first estimate ut by calculating the residuals from a long VAR-process
    B = VAR_LS(data, max(10, int(1.5 * (p + q))))[0]
    ut_est = VAR_residuals(data, B, p)

    Y = data[:, p:]
    X = np.ones((K * (p + q), T))
    Xt = np.zeros(K * (p + q))
    for t in range(p, T + p):
        for subt in range(p):
            start_i = subt * K
            stop_i = (subt + 1) * K
            Xt[start_i:stop_i] = data[:, t - subt - 1]
        for subt in range(p, p + q):
            start_i = subt * K
            stop_i = (subt + 1) * K
            Xt[start_i:stop_i] = ut_est[:, t - subt - p - 1]
        X[:, t - p] = Xt

    # R might not be necessary since we do not limit any parameters here
    R = np.identity(N)
    IK = np.identity(K)
    gamma = ((R.T @ np.linalg.inv(kron(X @ X.T, IK) @ R))
             @ R.T @ kron(X, IK) @ vec(Y))
    residuals_arma_vec = vec(Y) - kron(X.T, IK) @ R @ gamma
    residuals_arma = residuals_arma_vec.reshape((K, T), order="F")
    # make the residuals have the same length as the data
    residuals_arma = np.concatenate((np.zeros((K, p)), residuals_arma),
                                    axis=1)
    sigma_u_arma = residuals_arma @ residuals_arma.T / T
    AM = gamma.reshape((K, -1), order="F")
    # the following expression leads to the same result...
    # AM = Y * X.T * (X * X.T).I
    return AM, sigma_u_arma, residuals_arma


def VARMA_LS_sim(AM, p, q, sigma_u, means, T, S=None, m=None, ia=None,
                 m_trend=None, n_sim_multiple=2, fixed_data=None):
    """Generates a time series based on the VARMA-parameters AM.
    S and m should be sequences of length K. S is a variable-discerning
    multiplier and m a adder, respectively.

    Parameters
    ----------
    AM :       (K,K*(p+q)) array
               The parameters of the VARMA-process. The first p columns are
               interpreted as the Ai-matrices of the auto regressive part. The
               last q columns as the Mi-matrices of the moving average part. K
               is the number of variables simulated.
    p :        integer
               Order of the auto regressive process.
    q :        integer
               Order of the moving average process.
    sigma_u :  (K,K) array
               Covariance matrix of the residuals.
    means :    (K,) array
               Process means. Used as starting values.
    T :        integer
               Desired length of the output time series.
    S :        (K,K) array, optional
               Used as multiplicative change of the disturbance vector to
               increase the variance of the output.
    m :        (K,T) array_like, optional
               Used as additive change during simulation to increase mean of
               the output.
    n_sim_multiple : integer
                    Generate n_sim_multiple * T timesteps. Only the last T
                    timesteps will be returned.
    ia :       (K,T) array_like, optional
               Interannual variability. Used as an additive change during
               simulation to get time-dependent disturbances.
    m_trend :  (K,) array_like, optional
               Used as additive change gradient during simulation to increase
               mean of the output gradually.
    fixed_data : (K,T) array_like, optional
                Keeps the provided time-series fixed. Use np.nans to signify
                values that are not fixed.
                Can be used to simulate hierarchically.

    Returns
    -------
    out :    (K, T) ndarray
             K-dimensional simulated time series.

    """
    K = AM.shape[0]
    if S is None:
        S = np.identity(K, dtype=float)
    n_sim_steps = n_sim_multiple * T + p
    if m is None:
        m = np.zeros((K, n_sim_steps))
    else:
        m = _scale_additive(m, AM, p)
        # we have to expand m to include the pre-simulation timesteps
        m = np.tile(m, n_sim_multiple)
    m_trend = np.asarray([0] * K) if m_trend is None else np.asarray(m_trend)
    m_trend = _scale_additive(m_trend, AM, p)

    if ia is not None:
        ia = _scale_additive(ia, AM, p)

    # the first p columns are initial values, which will be omitted later
    Y = np.zeros((K, n_sim_steps))
    Y[:, :p] = means.reshape((K, -1))

    Y[:, -m.shape[1]:] += m
    start_t = Y.shape[1] - T
    ut = np.array([rng.multivariate_normal(K * [0], sigma_u)
                   for i in range(q)]).reshape((K, q))
    for t in range(p, n_sim_steps):
        # shift the old values back and draw a new random vector
        ut[:, :-1] = ut[:, 1:]
        ut[:, -1] = rng.multivariate_normal(K * [0], sigma_u)
        Y[:, t] = ut[:, -1][np.newaxis, :]

        # non-standard scenario stuff
        Y[:, t] = S @ Y[:, t]
        if t > start_t:
            if ia is not None:
                Y[:, t] += ia[:, t - start_t].T
            # apply changes as a trend
            Y[:, t] += float(t - start_t) / T * m_trend

        # conventional VARMA things
        for i in range(p):
            Ai = AM[:, i * K: (i + 1) * K]
            Y[:, t] += Ai @ Y[:, t - i - 1]
        for i in range(p, p + q):
            Mi = AM[:, i * K: (i + 1) * K]
            Y[:, t] += Mi @ ut[:, -1 - i + p]

        if (fixed_data is not None) and (t >= start_t):
            # fixing what's asked to be held constant
            Y[:, t] = \
                np.where(np.isnan(fixed_data[:, t - start_t]),
                         Y[:, t],
                         fixed_data[:, t - start_t])

        if fixed_data is None:
            Y[:, t] += means

    return Y[:, -T:]


def vec(A):
    """The vec operator stacks 2dim matrices into 1dim vectors column-wise.
    See p.661f.

    >>> A = np.arange(4).reshape(2, 2)
    >>> A
    array([[0, 1],
           [2, 3]])
    >>> vec(A)
    array([[0],
           [2],
           [1],
           [3]])
    """
    return A.T.ravel()[:, None]


def unvec(sequence, K):
    """The inverse of vec.

    >>> a = range(4)
    >>> a
    [0, 1, 2, 3]
    >>> unvec(a, K=2)
    array([[0, 2],
           [1, 3]])
    """
    return np.array(sequence).reshape(K, -1, order="F")


def vech(A):
    """The vech operator removes the upper triangular part of a matrix and
    returns the rest in a column-stacked form.
    See p.661f.

    >>> A = np.arange(4).reshape(2, 2)
    >>> A
    array([[0, 1],
           [2, 3]])
    >>> vech(A)
    array([[0],
           [2],
           [3]])
    """
    rows, columns = np.mgrid[0:A.shape[0], 0:A.shape[1]]
    return A.T[rows.T >= columns.T][:, None]


def unvech(sequence, K):
    """The inverse of vech.

    >>> a = np.array([0, 2, 3])
    >>> unvech(a, K=2)
    array([[ 0.,  2.],
           [ 2.,  3.]])
    """
    A = np.empty((K, K))
    A[np.tril_indices_from(A)] = np.squeeze(sequence)
    A[np.triu_indices_from(A, k=1)] = A[np.tril_indices_from(A, k=-1)]
    return A


def SC(sigma_u, p, T):
    """Schwarz criterion for VAR_LS order selection (p.150). To be minimized.
    """
    K = sigma_u.shape[0]
    return np.log(np.linalg.det(sigma_u)) + np.log(T) / T * p * K ** 2


def HQ(sigma_u, p, T):
    """Hannan-Quinn for VAR_LS order selection (p.150). To be minimized.
    """
    K = sigma_u.shape[0]
    return np.log(np.linalg.det(sigma_u)) + np.log(np.log(T)) / T * p * K ** 2


def AIC(sigma_u, p, T):
    """Akaike Information Criterion for order selection of a VAR process.
    See p.147"""
    K = sigma_u.shape[0]
    return np.log(np.linalg.det(sigma_u)) + old_div((2 * p * K ** 2), T)


def FPE(sigma_u, p, T):
    """Final prediction error.
    See p.147"""
    K = sigma_u.shape[0]
    return (((T + p * K + 1) / (T - p * K - 1)) ** K * np.linalg.det(sigma_u))


def VAR_order_selection(data, p_max=10, criterion=SC, estimator=VAR_LS,
                        est_kwds=None):
    """Order selection for VAR processes to allow parsimonious
    parameterization.

    Parameters
    ----------
    data : (K, T) ndarray
        Input data with K variables and T timesteps.
    p_max : int, optional
        Maximum number of autoregressive order to evaluate.
    criterion : function, optional
        Information criterion that accepts sigma_u, p, and T and returns
        something that gives small values for a parsimonious set of these
        parameters.

    Returns
    -------
    p : int
        Suggested autoregressive order.

    See also
    --------
    AIC : Akaike Information criterion
    FPE : Final Prediction Error
    HQ : Hannan-Quinn information criterion
    SC : Schwartz Criterion
    VAR_LS : Least-squares estimator.
    VAR_residuals : Returns the residuals based on given data and LS estimator
    VAR_LS_sim : Simulation based on LS estimator.
    VAR_LS_predict : Predict given prior data and LS estimator.

    """
    T = data.shape[1]
    if est_kwds is None:
        est_kwds = {}
    return np.argmin([criterion(estimator(data, p, **est_kwds)[1], p, T)
                      for p in range(p_max + 1)])


def VARMA_order_selection(data, p_max=5, q_max=5, criterion=SC,
                          plot_table=False, *args, **kwds):
    """Returns p and q, the orders of a VARMA process that allows for
    parsimonious parameterization.
    Naive extension of VAR_order_selection without a theoretical basis!"""
    K, T = data.shape
    sigma_us = np.nan * np.empty((p_max + 1, q_max + 1, K, K))
    # we ignore the cases where either p or q is 0, because VARMA_LS_prelim
    # chokes on that
    for p in range(1, p_max + 1):
        for q in range(q_max + 1):
            if q is 0:
                sigma_us[p, q] = VAR_LS(data, p, *args, **kwds)[1]
            sigma_us[p, q] = VARMA_LS_prelim(data, p, q, *args, **kwds)[1]

    crits = list(criterion)
    criterion_table = np.nan * np.empty((len(crits), p_max + 1, q_max + 1))
    for crit_i, crit in enumerate(crits):
        for p in range(1, p_max + 1):
            for q in range(q_max + 1):
                criterion_table[crit_i, p, q] = crit(sigma_us[p, q], p + q, T)

    if plot_table:
        for crit_i, crit in enumerate(crits):
            ts.matr_img(criterion_table[crit_i],
                        "Information criterion table. %s" % repr(crit))
            ts.plt.xlabel("q")
            ts.plt.ylabel("p")

    p_mins, q_mins = list(zip(*[np.unravel_index(np.nanargmin(criterion_table[ii]),
                                            criterion_table[ii].shape)
                           for ii in range(len(crits))]))
    return p_mins, q_mins, criterion_table


def _scale_additive(additive, A, p=None):
    """Scale an additive online component of a simulation. This prevents
    the overshooting due to auto- and crosscorrelations.

    Parameters
    ----------
    additive :   (K,) or (K,T) ndarray
                 Additive component to be scaled. K is the number of variables
                 simulated.Can be m, m_trend or ia of VARMA_LS_sim, for
                 example.
    A :          (K,K*p) ndarray
                 Parameters of the VAR process. p is the order of the VAR
                 process.
    p :          int, optional
                 Order of the VAR process. If given, only the first K*p columns
                 of A will be interpreted as the parameters of the VAR process.
                 Allows AM to be given as A, which also includes the VMA
                 parameters.

    Returns
    -------
    additive :   (K,) or (K,T) ndarray
                 Scaled additive component.
    """

    A = np.asarray(A)
    K = A.shape[0]
    if p is None:
        p = A.shape[1] / K
        if p * K != A.shape[1]:
            raise ValueError("A is not (K,K*p)-shape.")

    scale_matrix = np.identity(K)
    for i in range(p):
        scale_matrix -= A[:, i * K: (i + 1) * K]
    return scale_matrix @ additive


###############################################################################
## WARNING! The following functions were NOT tested thoroughly!!!!!!!!!!!!!!!!!
###############################################################################

def VAR_LS_predict(data_past, B, sigma_u, T=1, n_realizations=1):
    """Based on a least squares estimator, predict a time-series of the form
    ..math::y(t) = nu + A1*y(t-1) + ... + Ap*y(t-p) + ut
    B contains (nu, A1, ..., Ap).

    Parameters
    ----------
    data_past :      (K, p) ndarray
    B :              (K, p * K + 1) ndarray
                     Parameters of the VAR-process of order p.
    sigma_u :        (K, K) ndarray
                     Covariance matrix of the residuals of the VAR-process.
    T :              int
                     Number of timesteps to predict.
    n_realizations : int
                     Number of realizations. If > 1, gaussian disturbances
                     are added. So if n_realizations=1, the prediction is a
                     best guess.

    Returns
    -------
    Y :             (K, T) or (K, T, n_realizations) ndarray

    References
    ----------
    See p. 707f"""
    # number of variables
    K = B.shape[0]
    # order of VAR_LS-process
    p = (B.shape[1] - 1) / K
    nu = B[:, 0].ravel()

    # the first p columns are initial values, which will be omitted later
    Y = np.zeros((K, data_past.shape[1] + T, n_realizations))
    Y[:, :-T] = data_past[..., np.newaxis]

    for t in range(Y.shape[1] - T, Y.shape[1]):
        for r in range(n_realizations):
            Y[:, t, r] = nu
            if n_realizations > 1:
                Y[:, t, r] += rng.multivariate_normal(K * [0], sigma_u)

            for i in range(p):
                Ai = np.asarray(B[:, 1 + i * K: 1 + (i + 1) * K])
                Y[:, t, r] += np.squeeze(np.dot(Ai, Y[:, t - i - 1, r]))

    return np.squeeze(Y[:, -T:])


def VAR_YW(data, p=2):
    """Yule-Walker parameter estimation for a vector auto-regressive model of
    the form Y^0 = A*X + U
    Refer to p. 83ff.
    Here we assume that the data is already mean-adjusted!
    """
    # number of variables
    K, T = data.shape[0], data.shape[1] - p
    # Y is a (K, T) matrix
    Y = np.asmatrix(data[:, p:])

    # X is nearly the same as Z in VAR_LS, but without the first row of ones
    X = np.asmatrix(np.empty((K * p, T)))
    Xt = np.empty((K * p, 1))
    for t in range(p, T + p):
        for subt in range(p):
            # HACK! check the p
            Xt[subt * K:(subt + 1) * K] = data[:, t - subt - 1].reshape((K, 1))
        X[:, t - p] = Xt

    # A contains all the parameters (A1, ..., Ap)
    A = np.empty((K, K * p))
    Gamma_y = np.empty_like(A)
    # cov-matrices for up to p lags
    for lag in range(1, p + 1):
        # unbiased cross-covariance
        cov = ts.cross_cov(data, lag) / (T + p - lag)
        start_i = (lag - 1) * cov.shape[0]
        stop_i = lag * cov.shape[0]
        Gamma_y[:, start_i:stop_i] = cov
    Gamma_Y = np.empty((K * p, K * p))
    for ii in range(p):
        start_i = ii * K
        stop_i = (ii + 1) * K
        for jj in range(p):
            start_j = jj * K
            stop_j = (jj + 1) * K
            cov = ts.cross_cov(data, ii - jj) / (T + p - abs(ii - jj))
            Gamma_Y[start_i:stop_i, start_j:stop_j] = cov
    A = Gamma_y @ Gamma_Y.I

    # lets use the same noise as VAR_LS. no idea if this is justified
    sigma_u = Y @ Y.T - Y @ X.T @ np.linalg.inv(X @ X.T) @ X @ Y.T
    sigma_u /= T - K * p - 1

#    A_dash = Y * X.T * (X * X.T).I
#    matr_img(np.asarray(A), "A")
#    matr_img(np.asarray(A_dash), "A_dash")
#    plt.show()
    return A, sigma_u


def VAR_YW_sim(A, sigma_u, T):
    """Based on a Yule-Walker estimator, simulate a time-series of the form
    ..math:: y(t) = A_1*y(t-1) + ... + A_p*y(t-p) + u(t)
    A contains (A1, ..., Ap).
    See p. 707f"""
    # number of variables
    K = A.shape[0]
    # order of VAR-process
    p = old_div(A.shape[1], K)

    # the first p columns are initial values, which will be omitted later
    Y = np.zeros((K, T + p))

    for t in range(p, T + p):
        ut = rng.multivariate_normal(K * [0], sigma_u).reshape(K, 1)
        Y[:, t] = ut
        for i in range(p):
            Ai = A[:, i * K: (i + 1) * K]
            Y[:, t] += Ai @ Y[:, t - i]

    return Y[:, p:]


# def VAR_YW_residuals(data, A, p=2):
#    K, T = data.shape[0], data.shape[1] - p
#    resi = np.copy(data)
#    for t in xrange(p, T + p):
#        for i in range(p):
#            Ai = A[:, i * K: (i + 1) * K]
#            resi[:, t] -= \
#                np.squeeze(np.asarray(Ai * data[:, t - i - 1].reshape(K, 1)))
#    return resi


def _ut_gamma_part(data, p, q, AM, ut):
    """Recursive calculation of the partial derivatives del ut /del gamma.
    See Lemma 12.1 p.468"""
    K, T = data.shape
    N = K ** 2 * (p + q)

    Y = data[:, p:]
    R = np.identity(N)
    A_0 = np.identity(K)
    IK_zeros = np.zeros((K ** 2, N))
    IK_zeros[:K ** 2, :K ** 2] = np.identity(K ** 2)
    zero_IK = np.identity(N)
    ut_gamma_part = np.zeros((K, N, T + p))

    for t in range(p, T - p):
        varma = np.zeros((K, 1))
        for i in range(p):
            Ai = AM[:, i * K: (i + 1) * K]
            varma += Ai @ Y[:, t - i]
        for i in range(p, p + q):
            Mi = AM[:, i * K: (i + 1) * K]
            varma += Mi @ ut[:, -1 - i + p, np.newaxis]

        prev_yu = np.empty(K * (p + q))
        for i in range(p):
            prev_yu[i * K:(i + 1) * K] = Y[:, t - i].T
        for i in range(p, p + q):
            prev_yu[i * K:(i + 1) * K] = ut[:, t - i].T
        M_gamma_part = np.zeros((K, N))
        for i in range(p, p + q):
            Mi = AM[:, i * K: (i + 1) * K]
            M_gamma_part += Mi @ ut_gamma_part[..., t - i + p]
        ut_gamma_part[..., t] = \
            ((A_0 @ kron(varma.T, A_0.T)) @ IK_zeros @ R -
             kron(prev_yu, A_0.I) @ zero_IK @ R -
             A_0 @ M_gamma_part)
            # ((A_0.I * kron(varma.T, A_0.T)) * IK_zeros * R -
            #  kron(prev_yu, A_0.I) * zero_IK * R -
            #  A_0.I * M_gamma_part)
    return ut_gamma_part


def VARMA_LS(data, p, q, rel_change=1e-3):
    """Implementation of the scoring algorithm to fit a VARMA model. p.470ff"""
    AM_pre, sigma_u_pre = VARMA_LS_prelim(data, p, q)[:2]
    # do not trust the estimator of the residuals
    ut = VARMA_residuals(data, AM_pre, p, q)
    K, T = data.shape
    N = K ** 2 * (p + q)

    det_new = np.linalg.det(sigma_u_pre)
    # set det_old to something that will cause the while loop to execute at
    # least one time
    det_old = rel_change ** -1 * det_new
    print ("Determinant of preliminary residual covariance matrix: %f" %
           det_old)

    AM = AM_pre
    gamma = vec(AM)

    ts.matr_img(AM, "AM p=%d q=%d Preliminary" % (p, q))
    ii = 0
    while (det_new > 1e-60) and (old_div(np.abs(det_old - det_new), det_old)
                                 > rel_change):
        ut_gamma_part = _ut_gamma_part(data, p, q, AM, ut)
        sigma_u_gamma = T ** -1 * np.sum([ut[:, t] * ut[:, t].T  # np.newaxis]
                                          for t in range(ut.shape[1])],
                                         axis=0)
        sigma_u_gamma_inv = np.linalg.inv(sigma_u_gamma)
        # information matrix
        IM = np.sum([ut_gamma_part[..., t].T @ sigma_u_gamma_inv @
                     ut_gamma_part[..., t]
                     for t in range(T)], axis=0)
        likeli_gamma_part = np.sum([ut[:, t, np.newaxis].T @
                                    sigma_u_gamma_inv @ ut_gamma_part[..., t]
                                    for t in range(ut.shape[1])],
                                   axis=0)
        gamma -= np.linalg.inv(IM) @ likeli_gamma_part.T

        AM = gamma.reshape((K, K * (p + q)), order="F")
        ut = VARMA_residuals(data, AM, p, q)
        Y = data  # [:, p:])
        X = np.ones((K * (p + q), T))
        Xt = np.zeros((K * (p + q), 1))
        for t in range(p, T):
            for subt in range(p):
                start_i = subt * K
                stop_i = (subt + 1) * K
                Xt[start_i:stop_i] = data[:, t - subt - 1].reshape((K, 1))
            for subt in range(p, p + q):
                start_i = subt * K
                stop_i = (subt + 1) * K
                Xt[start_i:stop_i] = ut[:, t - subt - p - 1].reshape((K, 1))
            X[:, t - p] = Xt
        IK = np.identity(K)
        R = np.identity(N)
        gamma2 = (np.linalg.inv(R.T @ kron(X @ X.T, IK) @ R)
                  @ R.T @ kron(X, IK) @ vec(Y))
        residuals_arma_vec = vec(Y) - kron(X.T, IK) @ R @ gamma2
        residuals_arma = residuals_arma_vec.reshape((K, T), order="F")
        # make the residuals have the same length as the data
        ut = np.concatenate((np.zeros((K, p)), residuals_arma), axis=1)
        sigma_u = residuals_arma @ residuals_arma.T / T

        det_new, det_old = np.linalg.det(sigma_u), det_new
        print("Determinant of residual covariance matrix: %f" % det_new)
        ii += 1
        if ii > 1:
            ts.matr_img(np.asarray(gamma2.reshape((K, K * (p + q)),
                                                  order="F")),
                        "AM p=%d q=%d Iteration: %d" % (p, q, ii))

    AM = gamma2.reshape((K, K * (p + q)), order="F")
    return AM, sigma_u, ut


def VARMA_residuals(data, AM, p, q):
    K, T = data.shape[0], data.shape[1] - p
    resi = np.copy(data)
    for t in range(p, T + p):
        for i in range(p):
            Ai = AM[:, i * K: (i + 1) * K]
            resi[:, t] -= Ai @ data[:, t - i - 1].reshape(K, -1)
        for i in range(p, p + q):
            Mi = AM[:, i * K: (i + 1) * K]
            resi[:, t] -= Mi @ resi[:, t - i + p - 1].reshape(K, -1)
    return resi


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import tempfile
    import os
    import vg
    vg.conf = vg.vg_base.conf = vg.config_template
    p = 2
    T = 2 * 365
    var_names = (
        # we do not use precipitation here as long as we cannot
        # disaggregate it properly
        # "R",
        "theta", "Qsw", "ILWR", "rh",
        # "u", "v"
        )
    met_vg = vg.VG(var_names)
    met_vg.fit(p, extro=True)

