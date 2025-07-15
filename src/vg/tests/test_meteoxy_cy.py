import time
import numpy as np
import pandas as pd
import numpy.testing as npt
import vg
from vg.meteo import meteox2y, meteox2y_cy, brunner

import xarray as xr


class Test(npt.TestCase):
    def setUp(self):
        dtimes = pd.date_range("2000", "2001", freq="h")
        self.lat = 47.66
        self.lon = 9.18
        self.doys = vg.times.datetime2doy(dtimes)

    def test_pot_s_rad(self):
        time0 = time.perf_counter()
        qsw_2d = -1 * np.ones((22 - 4, len(self.doys)))
        for i, doy in enumerate(self.doys):
            date = ["%03d %02d" % (doy, hour) for hour in range(4, 22)]
            qsw_2d[:, i] = meteox2y.pot_s_rad(
                date, longt=self.lon, lat=self.lat, in_format="%j %H"
            )
        qsw_py = qsw_2d.sum(axis=0).ravel()
        time_py = time.perf_counter()

        qsw_2d = -1 * np.ones((22 - 4, len(self.doys)))
        for i, doy in enumerate(self.doys):
            date = ["%03d %02d" % (doy, hour) for hour in range(4, 22)]
            qsw_2d[:, i] = meteox2y_cy.pot_s_rad(
                date, longt=self.lon, lat=self.lat, in_format="%j %H"
            )
        qsw_cy = qsw_2d.sum(axis=0).ravel()
        time_cy = time.perf_counter() - time_py
        time_py -= time0
        print("Time python: ", time_py)
        print("Time cython: ", time_cy)
        npt.assert_almost_equal(qsw_cy, qsw_py)

    def test_sunshine_pot(self):
        time0 = time.perf_counter()
        hours_cy = meteox2y_cy.sunshine_pot(
            self.doys, lat=self.lat, longt=self.lon
        )
        time_cy = time.perf_counter()
        hours_py = meteox2y.sunshine_pot(
            self.doys, lat=self.lat, longt=self.lon
        )
        time_py = time.perf_counter() - time_cy
        time_cy -= time0
        print("Time python: ", time_py)
        print("Time cython: ", time_cy)
        npt.assert_almost_equal(hours_cy, hours_py)

    def test_brunner_compound(self):
        Ta = np.arange(10.0)
        P = Ta[::-1]
        hot_dry_cy = meteox2y_cy.brunner_compound(Ta, P)
        npt.assert_almost_equal(
            hot_dry_cy, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        hot_dry_py = brunner.brunner_compound(Ta, P)
        npt.assert_almost_equal(hot_dry_cy, hot_dry_py)

        rs = np.random.RandomState(0)
        Ta = xr.DataArray(
            rs.randn(2, 100, 10000), dims=["model", "station", "time"]
        )
        P = Ta + 0.5 * rs.randn(*Ta.shape)
        bc_py = brunner.brunner_compound(Ta, P, sequential=True, progress=True)
        bc_cy = meteox2y_cy.brunner_compound(Ta, P, progress=True)
        npt.assert_almost_equal(bc_py, bc_cy)

    def tearDown(self):
        pass


if __name__ == "__main__":
    npt.run_module_suite()
