from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import warnings

import numpy as np
from tqdm import tqdm
from scipy import optimize

try:
    import pyfftw
    from pyfftw.interfaces import numpy_fft as fftw
    FFTW = True
except ImportError:
    warnings.warn("Could not import pyfftw. Using the slower numpy functions.")
    FFTW = False
from vg.time_series_analysis import time_series


if FFTW:
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(5)
    N_THREADS = 8


class Spectral(object):

    def __init__(self, autocov, T, crange=None, cthresh=.01, sigma=1,
                 pool_size=None, verbose=False):
        """Simulate a univariate time series using FFT.
        This is an attempt to simplify Philips spectralsim to 1d.

        Parameter
        ---------
        autocov : callable
            Autocovariance model returning a covariance for a given distance.
        T : int
            Number of timesteps to simulate.
        crange : int or None, optional
            Correlation range of autocov. If `None`, it will be inferred from
            `autocov` and `cthresh`
        cthresh : float, optional
            Threshold for finding the correlation range.
        pool_size : None or int, optional
            Do not generate new realizations each time sim_n is called, but
            use a pool of realizations instead.
        verbose : boolean, optional

        Returns
        -------
        sim : (T,) float ndarray
            Simulated time series.
        """
        self.verbose = verbose
        self.pool_size = pool_size
        self._T = None
        self.T = T
        self.pool = None
        self.sigma = sigma

        if FFTW:
            self.ifft_func = \
                lambda *args, **kwds: fftw.ifft(threads=N_THREADS,
                                                *args, **kwds)
        else:
            self.ifft_func = np.fft.ifft
        if crange is None:
            # first, find out the maximum range
            # there is room for speed improvement here
            # distance from origin
            dists = np.arange(T)
            # covariances (in fourier space??)
            covs = autocov(dists)
            try:
                cutoff = np.min(np.where(old_div(covs, covs[0]) < cthresh)[0])
            except ValueError:
                # meaning, T < crange
                cutoff = T
        else:
            cutoff = crange

        size_old = T + cutoff
        # increase size to fit inside a power of two, in order to speed up fft
        size = 2 ** int(np.ceil(np.log2(size_old)))
        dists = np.arange(size)
        # "nonnegative definite embedding in circular matrices"
        dists = np.min((dists,
                        size - dists,
                        np.full_like(dists, size_old)),
                       axis=0)
        # recalculate covs to achieve periodic "grid"
        covs = autocov(dists)

        if FFTW:
            fft_covs = np.abs(fftw.fft(covs, threads=N_THREADS))
        else:
            fft_covs = np.abs(np.fft.fft(covs))
        # we need those in the sim-property
        # note that size != npoints in SpectralND
        self.size = self.npoints = size
        # eigenvalues of decomposition (??)
        self.sqrt_fft_covs = np.sqrt(old_div(fft_covs, self.npoints))
        self.mslice = slice(int(old_div((size - T), 2)),
                            int(old_div((size + T), 2)))
        if pool_size:
            self.pool = self.sim_n(pool_size)

    @property
    def output_size(self):
        if self.T:
            return self.T,
        else:
            return 0

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if self.T is not None and value < self.T:
            try:
                self.pool = self.pool[..., :value]
            except AttributeError:
                # we don't have a pool yet. don't worry
                pass
        self._T = value

    @property
    def sim(self):
        real = np.random.normal(size=np.ravel(self.size))
        imag = np.random.normal(size=np.ravel(self.size))
        epsilon = real + 1j * imag
        rand = epsilon * self.sqrt_fft_covs
        return ((np.real(self.ifft_func(rand)) * self.npoints)[self.mslice] *
                self.sigma)

    def sim_white(self, epsilon):
        rand = epsilon * self.sqrt_fft_covs
        return ((np.real(self.ifft_func(rand)) * self.npoints)[self.mslice] *
                self.sigma)

    def sim_n(self, n):
        """Return (n x *size) array of simulations.  Will return a random
        sample of a pool, if a pool_size was given in initialization.
        The pool will grow, if n > pool_size.
        """
        def gen_n(extra_n):
            if self.verbose:
                prog = tqdm(extra_n)
            sims = np.empty((extra_n,) + self.output_size)
            for sim_i in range(extra_n):
                sims[sim_i] = self.sim
                if self.verbose:
                    prog.update(sim_i + 1)
            if self.verbose:
                prog.close()
            return sims

        if self.pool_size is None:
            return gen_n(n)
        else:
            if self.pool is None:
                self.pool = gen_n(n)
            if n > self.pool_size:
                self.pool = np.concatenate((self.pool,
                                            gen_n(n - self.pool_size)))
                self.pool_size = n
            return self.pool[np.random.randint(self.pool_size, size=n)]


class SpectralND(Spectral):

    """A clean-up re-write of philips old code."""

    def __init__(self, cov, domainshape=(100, 100), crange=None, cthresh=.01,
                 sigma=1, pool_size=None, verbose=False, scale=1.):
        """
        Parameter
        ---------
        cov : callable
            Autocovariance model returning a covariance for a given distance.
        domainshape : tuple of len ndim
            Number of grid points in all directions
        crange : int or None, optional
            Correlation range of autocov. If `None`, it will be inferred from
            `cov` and `cthresh`
        cthresh : float, optional
            Threshold for finding the correlation range.
        sigma : float, optional
            Standard deviation of the field (experimental)
        pool_size : None or int, optional
            Do not generate new realizations each time sim_n is called, but
            use a pool of realizations instead.
        scale : float or sequence of floats, optional
            Dimension scale in general or corresponding to every dimension.
        verbose : boolean, optional
        """
        self.verbose = verbose
        self.domainshape = domainshape = np.asarray(domainshape)
        self.size = domainshape.size
        self.sigma = sigma
        self.pool_size = pool_size
        self.pool = None
        ndim = len(domainshape)

        scale = np.atleast_1d(scale)
        if len(scale) == 1:
            scale = np.repeat(scale, ndim)

        if FFTW:
            self.ifft_func = \
                lambda *args, **kwds: fftw.ifftn(threads=N_THREADS,
                                                 *args, **kwds)
        else:
            self.ifft_func = np.fft.ifftn

        if crange is None:
            root_func = lambda h: cthresh - cov(h)
            maxdist = np.sqrt(np.sum((dim * dim_scale) ** 2
                                     for dim, dim_scale
                                     in zip(domainshape, scale)))
            try:
                crange = optimize.brentq(root_func, 0., maxdist)
            except ValueError:
                crange = .5 * maxdist
                warnings.warn("Cannot determine correlation length. "
                              "I will use %.3f" % crange)
            cutoff = int(np.ceil(old_div(crange, np.max(scale))))
        else:
            cutoff = crange

        size_old = domainshape + cutoff
        # increase size to fit inside a power of two, in order to speed up fft
        domainshape_larger = 2 ** np.ceil(np.log2(size_old)).astype(int)
        grid = np.mgrid[[slice(0, dim_scale * dim, dim_scale)
                         for dim, dim_scale
                         in zip(domainshape_larger, scale)]]
        for _ in range(ndim):
            domainshape_larger = domainshape_larger[:, np.newaxis]
        grid = np.min((grid, domainshape_larger - grid), axis=0)

        # distances to origin
        origin_dists = np.sqrt((grid ** 2).sum(axis=0))
        # covariances in fourier space
        covs = cov(origin_dists)
        if FFTW:
            fft_covs = np.abs(fftw.fftn(covs, threads=N_THREADS))
        else:
            fft_covs = np.abs(np.fft.fftn(covs))
        # eigenvalues of decomposition
        self.npoints = np.prod(domainshape_larger)
        self.sqrt_fft_covs = np.sqrt(old_div(fft_covs, self.npoints))
        # needed in self.sim to cut a piece of domainshape size out of
        # the generated, larger field
        self.mslice = [slice(dim) for dim in domainshape]
        # the size of the field we are generating (larger in hope of
        # acceleration)
        self.size = domainshape_larger
        if pool_size:
            self.pool = self.sim_n(pool_size)

    @property
    def output_size(self):
        return tuple(self.domainshape)


class MultiSpectral(Spectral):

    """Simulate multivariate time series."""

    def __init__(self, autocovs, cov, T, sigma=1, pool_size=None,
                 verbose=False):
        """Simulate multivariate time series with different autocovariances.

        Parameter
        ---------
        autocovs : sequence of callables of length K or ndarray
            Autocovariance models returning a covariance for a given distance.
            If an array is given, the autocovariances are estimated from that
            data.
        cov : (K, K) or (K, S) ndarray
            Covariance matrix. If given as a (K, S) array, the covariances will
            be estimated from that data. S is the number of timesteps in the
            input.
        T : int
            Number of timesteps to simulate.
        pool_size : None or int, optional
            Do not generate new realizations each time sim_n is called, but
            use a pool of realizations instead.
        verbose : boolean, optional

        Returns
        -------
        sim : (K, T) float ndarray
            Simulated time series.
        """
        # household variable to store and set T in the univariate Spectral
        # instances
        # self._T = T
        self.verbose = verbose
        self.pool_size = pool_size
        self.pool = None
        self.K = K = len(autocovs)
        self._T = None
        self.T = T
        self.sigma = sigma
        if type(autocovs) is np.ndarray and not callable(autocovs[0]):
            data = autocovs
            # the 'i=0' prevents i from being set to K for all iterations in
            # the list comprehension below
            ac = lambda i=0: lambda lag: time_series.auto_cov(data[i], lag)
            autocovs = [ac(i) for i in range(K)]
        if cov.shape[1] > K:
            cov = np.cov(cov)
        self.specs = [Spectral(autocov, T, pool_size=pool_size)
                      for autocov in autocovs]
        # sigma = np.array([np.sqrt(autocov(0)) for autocov in autocovs])
        self.sigma = sigma = np.sqrt(np.diag(cov))
        sigma_xy = sigma[:, None] * sigma[None, :]
        try:
            self.cholesky = np.linalg.cholesky(cov / sigma_xy)
        except np.linalg.LinAlgError:
            warnings.warn("LinAlgError occured. Fiddling with the " +
                          "Covariance Matrix now!")
            self.cholesky = np.linalg.cholesky(cov / sigma_xy +
                                               np.diag(K * [1e-9]))
        if pool_size:
            self.pool = self.sim_n(pool_size)
        else:
            self.pool = None

    @property
    def output_size(self):
        return self.K, self.T

    @property
    def sim(self):
        data = np.array([spec.sim for spec in self.specs])
        stds = data.std(axis=1)[:, None]

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2)
        # data2 = np.dot(self.cholesky, data / stds) * stds
        # vmin = min(np.cov(data).min(), np.cov(data2).min())
        # vmax = max(np.cov(data).max(), np.cov(data2).max())
        # axs[0].matshow(np.cov(data), vmin=vmin, vmax=vmax)
        # axs[1].matshow(np.cov(data2), vmin=vmin, vmax=vmax)
        # plt.show()

        data /= stds
        return np.dot(self.cholesky, data) * self.sigma[:, None]  # stds


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cov_model = lambda h: np.exp(-h)
    domainshape = 100, 100
    spec = SpectralND(cov_model, domainshape)
    plt.matshow(spec.sim)
    plt.show()
