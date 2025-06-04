"""
Created on 06.09.2012

@author: dirk
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import TestCase, assert_almost_equal

import vg
from vg.time_series_analysis import time_series


class Test(TestCase):
    def setUp(self):
        self.verbose = False
        self.data = np.array(
            [
                47,
                64,
                23,
                71,
                38,
                64,
                55,
                41,
                59,
                48,
                71,
                35,
                57,
                40,
                58,
                44,
                80,
                55,
                37,
                74,
                51,
                57,
                50,
                60,
                45,
                57,
                50,
                45,
                25,
                59,
                50,
                71,
                56,
                74,
                50,
                58,
                45,
                54,
                36,
                54,
                48,
                55,
                45,
                57,
                50,
                62,
                44,
                64,
                43,
                52,
                38,
                59,
                55,
                41,
                53,
                49,
                34,
                35,
                54,
                45,
                68,
                38,
                50,
                60,
                39,
                59,
                40,
                57,
                54,
                23,
            ],
            dtype=float,
        )
        self.sim = self.gen_time_series(rho=0.9)

    def gen_time_series(self, rho):
        # time series with known autocorrelation
        self.T = int(1e5)
        sim = vg.rng.normal(size=self.T)
        for tt in range(self.T):
            sim[tt] += rho * sim[tt - 1]
        return sim

    def test_autocorr_ar(self):
        """compare calculated autocorr on series with known autocorr."""
        phi = time_series.auto_corr(self.sim, 1)
        assert_almost_equal(0.9, phi, decimal=2)

    def test_autocorr_ar_multi(self):
        """compare calculated autocorr on series with known autocorr 2dim."""
        phi = time_series.auto_corr([self.sim, self.gen_time_series(0.6)], 1)
        assert_almost_equal([0.9, 0.6], phi, decimal=2)

    def test_autocorr_ar_nans(self):
        """compare calculated autocorr on series with known autocorr with nans."""
        sim = np.copy(self.sim)
        sim[:: int(self.T / 5)] = np.nan
        sim[1 :: int(self.T / 5)] = np.nan
        phi = time_series.auto_corr(sim, 1)
        assert_almost_equal(0.9, phi, decimal=2)

    def test_autocorr_ar_nans_multi(self):
        """compare calculated autocorr on series with known autocorr with nans
        2dim.
        """
        sim = np.copy([self.sim, self.gen_time_series(0.6)])
        sim[0, :: int(self.T / 3)] = np.nan
        sim[:, 1 :: int(self.T / 5)] = np.nan
        phi = time_series.auto_corr(sim, 1)
        assert_almost_equal([0.9, 0.6], phi, decimal=2)

    def test_autocorr_plt_acorr(self):
        """do we get the same as matplotlib.pyplot.acorr?"""
        corr_exp = plt.acorr(self.sim, maxlags=5)[1]
        corr = time_series.auto_corr(self.sim, range(-5, 6))
        assert_almost_equal(corr_exp, corr, decimal=4)

    def test_autocorr_box1(self):
        """see p.30f Time Series Analysis Box Jenkins"""
        phi = time_series.auto_corr(self.data[:10], 1)
        assert_almost_equal(phi, -0.79, decimal=2)

    def test_autocorr_box2(self):
        """see p.31f Time Series Analysis Box Jenkins"""
        rk = np.array(
            [
                -0.39,
                0.30,
                -0.17,
                0.07,
                -0.10,
                -0.05,
                0.04,
                -0.04,
                -0.00,
                0.01,
                0.11,
                -0.07,
                0.15,
                0.04,
                -0.01,
            ]
        )
        phis = np.array(
            [time_series.auto_corr(self.data, k) for k in range(1, 16)]
        )
        if self.verbose:
            for ii, (r, phi) in enumerate(zip(rk, phis)):
                print("%d %+.4f %+.4f" % (ii + 1, r, phi))
        assert_almost_equal(rk, phis, decimal=2)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_auto_corr']
    # run_module_suite()
    pass
