from __future__ import division
from builtins import range
from past.utils import old_div
import datetime
import numpy as np
from scipy import integrate
import numpy.testing as npt
import vg
from vg.time_series_analysis import _kde as kde
from vg import times


class Test(npt.TestCase):
    def setUp(self):
        n_years = 2
        # construct n_years of hourly data
        self.dtimes = np.array(
            [
                datetime.datetime(2000, 1, 1) + t * datetime.timedelta(hours=1)
                for t in np.arange(n_years * 365 * 24)
            ]
        )
        x = np.arange(len(self.dtimes))
        # this is roughly a daily cycle within a yearly cycle
        # the strength of the daily cycle is dependent on the doy
        self.noise = vg.rng.normal(size=len(x))
        self.data = (
            15 * -np.cos(x * 2 * np.pi / (365 * 24))
            + 5
            * (1 + np.cos(x * 2 * np.pi / (365 * 24)))
            * -np.cos(x * 2 * np.pi / 24)
            + 1.5 * self.noise
        )
        self.circ = circ = 1.5
        self.dists_exp = [
            0.0,
            ((6.0 / 24) ** 2 + (circ * 6 / 24.0) ** 2) ** 0.5,
            ((12.0 / 24) ** 2 + (circ * 12 / 24.0) ** 2) ** 0.5,
            ((18.0 / 24) ** 2 + (circ * 6 / 24.0) ** 2) ** 0.5,
            1.0,
            ((30.0 / 24) ** 2 + (circ * 6 / 24.0) ** 2) ** 0.5,
            ((36.0 / 24) ** 2 + (circ * 12 / 24.0) ** 2) ** 0.5,
            ((42.0 / 24) ** 2 + (circ * 6 / 24.0) ** 2) ** 0.5,
        ]
        doys = times.datetime2doy(self.dtimes[: 2 * 24 : 6])
        n_dist = len(self.dists_exp)
        self.dists_exp2 = [
            [
                kde.doyhour_distance(
                    doys[i], doys[j], doy_width=1, circ=self.circ
                )
                for i in range(n_dist)
            ]
            for j in range(n_dist)
        ]

    def tearDown(self):
        pass

    def test_kernel_integration(self):
        vg.reseed(0)
        sample = vg.rng.normal(size=1000)
        q1, q2 = 0.25, 0.75
        x0, x1 = map(lambda q: np.percentile(sample, 100 * q), (q1, q2))
        kernel_width = kde.optimal_kernel_width(sample)

        def density_func(x):
            return kde.kernel_density(kernel_width, sample, eval_points=x)

        integral = integrate.romberg(density_func, x0, x1)

        # import matplotlib.pyplot as plt
        # xx = np.linspace(x0, x1, 500)
        # densities = density_func(xx)
        # xx2 = np.linspace(sample.min(), sample.max(), 100)
        # densities2 = density_func(xx2)
        # integrals2 = integrate.cumtrapz(y=densities2, x=xx2, initial=0)
        # sample_sorted = sample.copy()
        # sample_sorted.sort()
        # sample_ranks = (np.arange(1., len(sample) + 1.) - .5) / len(sample)
        # plt.plot(xx2, densities2, label="kde all")
        # plt.plot(xx, densities, label="kde test")
        # plt.plot(xx2, integrals2, label="kde cdf all")
        # plt.plot(sample_sorted, sample_ranks, label="sample cdf")
        # plt.scatter(sample, np.zeros_like(sample), label="sample")
        # plt.legend(loc="best")
        # plt.show()

        npt.assert_almost_equal(integral, q2 - q1, decimal=3)

    def test_log_kernel_integration(self):
        sample = np.exp(np.linspace(0, 1, 500))
        vg.reseed(0)
        kernel_width = kde.optimal_kernel_width(np.log(sample))
        q1, q2 = 1e-3, 1 - 1e-3
        x0, x1 = map(lambda q: np.percentile(sample, 100 * q), (q1, q2))

        def density_func(x):
            density = kde.kernel_density(
                kernel_width, np.log(sample), eval_points=np.log(x)
            )
            return density / x

        xx = np.linspace(x0, x1 + 5 * kernel_width, 1000)
        densities = density_func(xx)
        integral = integrate.trapz(y=densities, x=xx)
        npt.assert_almost_equal(integral, q2 - q1, decimal=3)

    def test_distance_array_sparse(self):
        """Do we get the same as with distance_array?"""
        x_vec = np.arange(10)
        dists_exp = x_vec[None, :] - x_vec[:, None]
        mask = dists_exp < 0
        dists_exp[mask] = 0
        dists_act = kde.distance_array_sparse(x_vec, x_vec, mask)
        npt.assert_almost_equal(dists_exp, dists_act.toarray())
        # kde.distance_array_sparse.clear_cache()

    def test_apply_sparse_kernel(self):
        """Does sparse output the same as dense?"""
        data = np.arange(50)
        width = 5.0
        mask = np.full((len(data), len(data)), True, dtype=bool)
        mask_mask = (slice(None, None, 2), slice(None, None, 4))
        mask_sparse = np.copy(mask)
        mask_sparse[mask_mask] = False
        dens_exp = kde.apply_kernel(width, data)
        dens_exp_sparse = np.copy(dens_exp)
        dens_exp_sparse[~mask_sparse] = 0.0
        for kernel in (kde.gaussian_kernel, kde.gaussian_kernel_ne):
            # let's start with all unmasked points
            dens_act = kde.apply_sparse_kernel(kernel, width, data, mask)
            npt.assert_almost_equal(dens_exp, dens_act.toarray())
            kde.apply_sparse_kernel.clear_cache()
            # some masked points
            dens_act = kde.apply_sparse_kernel(
                kernel, width, data, mask_sparse
            )
            npt.assert_almost_equal(dens_exp_sparse, dens_act.toarray())
            kde.apply_sparse_kernel.clear_cache()

    def test_doyhour_distance_dt(self):
        dtimes = self.dtimes[: 2 * 24 : 6]
        from_time = self.dtimes[0]
        dists = kde.doyhour_distance_dt(from_time, dtimes, circ=self.circ)
        npt.assert_almost_equal(dists, self.dists_exp)

    def test_doyhour_distance(self):
        doys = kde.times.datetime2doy(self.dtimes[: 2 * 24 : 6])
        from_time = doys[0]
        dists = kde.doyhour_distance(
            from_time, doys, doy_width=1, circ=self.circ
        )
        npt.assert_almost_equal(dists, self.dists_exp)
        dists = kde.doyhour_distance(doys, doys, doy_width=1, circ=self.circ)
        npt.assert_almost_equal(dists, self.dists_exp2)
        dist = kde.doyhour_distance(1.0, 365.0, 1.0, 1.0)
        npt.assert_almost_equal(dist, 1.0)
        dist = kde.doyhour_distance(1.0, 365.0 + old_div(23.0, 24), 1.0, 1.0)
        npt.assert_almost_equal(dist, np.sqrt(2 * (old_div(1.0, 24)) ** 2))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    # import warnings
    # warnings.simplefilter("error", RuntimeWarning)
    npt.run_module_suite()
