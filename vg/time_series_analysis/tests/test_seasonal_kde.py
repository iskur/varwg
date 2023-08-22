from __future__ import division

import datetime

import numpy as np
import numpy.testing as npt
from builtins import range
from numpy.testing import TestCase, run_module_suite
from past.utils import old_div

from vg import times
from vg.time_series_analysis import (
    _kde as kde,
    distributions,
    seasonal_kde as skde,
)


class Test(TestCase):
    # def setUp(self):
    #     n_years = 2
    #     # construct n_years of hourly data
    #     self.dtimes = np.array([datetime.datetime(2000, 1, 1) +
    #                             t * datetime.timedelta(hours=1)
    #                             for t in np.arange(n_years * 365 * 24)])
    #     x = np.arange(len(self.dtimes))
    #     # this is roughly a daily cycle within a yearly cycle
    #     # the strength of the daily cycle is dependent on the doy
    #     self.noise = np.random.randn(len(x))
    #     self.data = (15 * -np.cos(x * 2 * np.pi / (365 * 24)) +
    #                  5 * (1 + np.cos(x * 2 * np.pi / (365 * 24))) *
    #                  -np.cos(x * 2 * np.pi / 24) +
    #                  1.5 * self.noise)
    #     self.circ = circ = 1.5
    #     self.dists_exp = [0.,
    #                       ((old_div(6., 24)) ** 2 + (circ * 6 / 24.) ** 2) ** .5,
    #                       ((old_div(12., 24)) ** 2 + (circ * 12 / 24.) ** 2) ** .5,
    #                       ((old_div(18., 24)) ** 2 + (circ * 6 / 24.) ** 2) ** .5,
    #                       1.,
    #                       ((old_div(30., 24)) ** 2 + (circ * 6 / 24.) ** 2) ** .5,
    #                       ((old_div(36., 24)) ** 2 + (circ * 12 / 24.) ** 2) ** .5,
    #                       ((old_div(42., 24)) ** 2 + (circ * 6 / 24.) ** 2) ** .5
    #                       ]
    #     doys = skde.times.datetime2doy(self.dtimes[:2 * 24:6])
    #     n_dist = len(self.dists_exp)
    #     self.dists_exp2 = [[kde.doyhour_distance(doys[i], doys[j],
    #                                              doy_width=1,
    #                                              circ=self.circ)
    #                         for i in range(n_dist)]
    #                        for j in range(n_dist)]

    def tearDown(self):
        pass

    # def test_seasonalhourlykde(self):
    #     "General test by fitting on synthetic data."""
    #     import vg, config_konstanz
    #     conf = config_konstanz
    #     vg.vg_base.conf = conf
    #     met_vg = vg.VG(("theta", "ILWR", "rh", "u", "v"))
    #     # from vg.meteo import avrwind
    #     # data = avrwind.component2angle(met_vg.met["u"], met_vg.met["v"])[1]
    #     thumb = False
    #     varname = "theta"
    #     conf.hpd = 1
    #     data = met_vg.met[varname][-(4 * 8760 + 24):]
    #     dtimes = met_vg.times_orig[-(4 * 8760 + 24):]
    #     sea_kde = skde.SeasonalHourlyKDE(data, dtimes, doy_width=5,
    #                                      fixed_pars=conf.par_known[varname],
    #                                      verbose=True)
    #     # sea_kde = skde.SeasonalHourlyKDE(self.data, self.dtimes,
    #     #                                      doy_width=15, verbose=True)
    #     # solution = kernel_widths, circs, quantile_grid = sea_kde.fit()  # order=3)
    #     # solution = kernel_widths, quantile_grid = sea_kde.fit()  # order=3)
    #     solution = kernel_widths, quantile_grid = sea_kde.fit(thumb, order=10)

    #     kernel_widths_padded = sea_kde._pad_torus(kernel_widths)
    #     kernel_widths_unpadded = sea_kde._unpad_torus(kernel_widths_padded)
    #     npt.assert_almost_equal(kernel_widths, kernel_widths_unpadded)

    #     # import matplotlib.pyplot as plt
    #     # if not thumb:
    #     #     np.savez("hourlykde_test_solutions",
    #     #              kernel_widths=kernel_widths)
    #     #     fig, ax = plt.subplots(1)
    #     #     ca = ax.matshow(kernel_widths)
    #     #     plt.colorbar(ca, orientation="horizontal")

    #     # sea_kde.scatter_pdf(solution, plot_kernel_width=False,
    #     #                     s_kwds=dict(s=30))

    #     # quantiles = sea_kde.cdf(solution, x=data, doys=met_vg.data_doys_raw)
    #     # data_trans = distributions.norm.ppf(quantiles)
    #     # fig, ax = plt.subplots(2, sharex=True)
    #     # ax[0].plot(dtimes, data)
    #     # ax[1].plot(dtimes, data_trans)
    #     # for a in ax:
    #     #     a.grid(True)
    #     # plt.show()
    #     # this tests whether the quantiles attained from qq-mapping
    #     # are uniformly distributed
    #     # npt.assert_array_less(.05, sea_kde.chi2_test())

    def test_fft2par(self):
        fft_pars = np.array([1 + 0j, 2 + 1j])
        dtimes = np.array(
            [
                datetime.datetime(2000, 12, 30)
                + t * datetime.timedelta(hours=1)
                for t in np.arange(30 * 24)
            ]
        )
        doys = times.datetime2doy(dtimes)
        trans = skde.fft2par(fft_pars, doys)
        npt.assert_equal(len(trans), len(doys))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    # import warnings
    # warnings.simplefilter("error", RuntimeWarning)
    run_module_suite()
