from pathlib import Path
import shutil
import tempfile
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from scipy.stats import distributions as sp_dists
from vg.time_series_analysis import (
    seasonal_distributions as sdists,
    distributions as dists,
)
from vg.meteo import meteox2y
import dwd_opendata
from vg import helpers as my
from vg import shelve
import vg

# script_home = os.path.dirname(vg.__file__)
# met_file = os.path.join(script_home, "sample.met")

longitude = 47.66
latitude = 9.18


def max_sunshine_hours(doys):
    from vg import times

    dates = times.doy2datetime(doys)
    sun_hours = meteox2y.sunshine_hours(
        dates,
        longitude=longitude,
        latitude=latitude,
        # does not matter, because we
        # are not interested in when,
        # but in how long
        tz_offset=0,
    )
    return sun_hours * 60


class Test(npt.TestCase):
    def setUp(self):
        self.verbose = True
        self.cache_dir = Path(tempfile.mkdtemp("vg_test"))

        theta_xr = dwd_opendata.load_station(
            "Konstanz", "air_temperature"
        ).squeeze()
        theta_xr = (
            theta_xr.sel(time=slice("2000", "2016"))
            .resample(time="D")
            .mean()
            .interpolate_na("time")
        )
        self.theta_data = theta_xr.values.squeeze()
        self.dt = theta_xr.time.to_dataframe().index.to_pydatetime()

    def tearDown(self):
        shutil.rmtree(self.cache_dir)

    def test_serialization(self):
        sdist_orig = sdists.SlidingDist(
            sp_dists.exponnorm, self.theta_data, self.dt, verbose=self.verbose
        )
        sh = shelve.open(str(self.cache_dir / "seasonal_cache_file"), "c")
        sh["theta"] = sdist_orig
        sh.close()
        sh = shelve.open(str(self.cache_dir / "seasonal_cache_file"), "c")
        sdist_shelve = sh["theta"]
        assert not my.recursive_diff(
            None,
            sdist_orig,
            sdist_shelve,
            verbose=True,
            plot=True,
            ignore_types=(sp_dists.rv_continuous,),
        )

    def test_rainmix(self):
        # prec_data = self.data[self.var_names.index("R")]
        prec_xr = dwd_opendata.load_station("Freiburg", "precipitation")
        prec_xr = (
            prec_xr.sel(time=slice("2000", "2016"))
            .resample(time="D")
            .sum()
            .interpolate_na("time")
        )
        prec_data = np.squeeze(prec_xr.values)
        dt = prec_xr.time.to_dataframe().index.to_pydatetime()
        threshold = 0.001 * 24
        dist = dists.RainMix(
            dists.kumaraswamy,
            # q_threshold=.9,
            threshold=threshold,
        )
        # dist.debug = True
        fixed_pars = dict(
            u=(lambda x: np.ones_like(x)), l=(lambda x: np.zeros_like(x))
        )
        sdist = sdists.SlidingDist(
            dist,
            prec_data,
            dt,
            # doy_width=int(365 / 8),
            doy_width=15,
            verbose=self.verbose,
            fixed_pars=fixed_pars,
            tabulate_cdf=True,
            # fft_order=100,
        )
        sol = sdist.fit()
        qq = sdist.cdf(sol)
        prec_recovered = sdist.ppf(sol, qq)
        assert all(prec_recovered >= 0)
        rain_mask = prec_data > threshold
        try:
            npt.assert_allclose(
                prec_recovered[rain_mask],
                prec_data[rain_mask],  # atol=0.152
            )
        except AssertionError:
            if self.verbose:
                fig, axs = plt.subplots(
                    nrows=1, ncols=2, width_ratios=(0.8, 0.2)
                )
                axs[0].plot(dt[rain_mask], prec_data[rain_mask], label="data")
                axs[0].plot(
                    dt[rain_mask], prec_recovered[rain_mask], label="recovered"
                )
                axs[1].set_aspect("equal")
                axs[1].scatter(
                    prec_data[rain_mask], prec_recovered[rain_mask], marker="x"
                )
                axs[1].set_xlabel("data")
                axs[1].set_ylabel("recovered")
                nonequal_ii = np.where(
                    ~np.isclose(
                        prec_data[rain_mask], prec_recovered[rain_mask]
                    )
                )[0]
                if len(nonequal_ii) < 100:
                    for nonequal_i in nonequal_ii:
                        axs[0].axvline(dt[rain_mask][nonequal_i])

                prec_sample = sdist.ppf(sol, vg.rng.uniform(0, 1, len(qq)))
                import xarray as xr

                prec_sample_xr = xr.DataArray(
                    prec_sample, coords=dict(time=dt), dims=["time"]
                )
                fig, axs = plt.subplots(
                    nrows=3, ncols=4, constrained_layout=True
                )
                axs = np.ravel(axs)
                ax_bins = {}
                for month_i, monthly in prec_sample_xr.groupby("time.month"):
                    ax = axs[month_i - 1]
                    bins = ax.hist(
                        monthly[monthly > threshold],
                        histtype="step",
                        label="sim",
                    )[1]
                    ax_bins[month_i] = bins
                for month_i, monthly in prec_xr.groupby("time.month"):
                    ax = axs[month_i - 1]
                    data = np.squeeze(monthly.values)
                    ax.hist(
                        data[data > threshold],
                        histtype="step",
                        label="obs",
                        bins=ax_bins[month_i],
                    )

                sdist.plot_seasonality_fit(sol)
                sdist.plot_monthly_fit(sol)
                sdist.plot_monthly_fit(dists_alt=dists.rainmix_kumaraswamy)

                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.hist(qq[np.isfinite(qq)], 40, histtype="step")
                ax.axvline(sdist.dist.q_thresh, color="k")
                plt.show()
            raise

    def test_rainmix_sun(self):
        import xarray as xr

        sun_xr = dwd_opendata.load_station("Konstanz", "sun").squeeze()
        dt_hourly = sun_xr.time.to_dataframe().index.to_pydatetime()
        max_minutes = meteox2y.max_sunshine_minutes(
            dt_hourly, longitude, latitude
        )
        sun_xr = xr.where(sun_xr > max_minutes, 0, sun_xr)
        sun_xr = (
            sun_xr.sel(time=slice("2000", "2016"))
            .resample(time="D")
            .mean()
            .interpolate_na("time")
        )
        sun_data = np.squeeze(sun_xr.values)
        dt = sun_xr.time.to_dataframe().index.to_pydatetime()
        # threshold = .01 * 24
        # threshold = .0005
        dist = dists.RainMix(
            dists.beta,
            q_thresh_lower=0.8,
            q_thresh_upper=0.975,
        )
        dist.debug = True
        fixed_pars = dict(u=max_sunshine_hours, l=(lambda x: np.zeros_like(x)))
        sdist = sdists.SlidingDist(
            dist,
            sun_data,
            dt,
            # doy_width=int(365 / 8),
            doy_width=15,
            verbose=True,
            fixed_pars=fixed_pars,
            tabulate_cdf=True,
        )
        sol = sdist.fit()
        # sdist.plot_monthly_fit(sol)
        # sdist.plot_monthly_fit()
        qq = sdist.cdf(sol)
        sun_back = sdist.ppf(sol, qq)
        over_thresh = sun_back > sdist.dist.thresh
        try:
            npt.assert_allclose(
                sun_back[over_thresh], sun_data[over_thresh], atol=0.0031
            )
        except AssertionError:
            plt.scatter(sun_data, sun_back)
            plt.show()
            raise
        # qq = np.full_like(qq, 1.0)
        # sun_hell = sdist.ppf(sol, qq)
        # sun_theo = max_sunshine_hours(times.datetime2doy(dt))
        # npt.assert_almost_equal(sun_hell, sun_theo)
        # assert np.all(sun_hell <= sun_theo)
        # npt.assert_array_less(sun_hell, sun_theo)
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.hist(qq, 40, histtype="step")
        # ax.axvline(sdist.dist.q_thresh, color="k")
        # plt.show()

    def test_cdf_table(self):
        # sdist_notable = sdists.SlidingDist(
        #     sp_dists.exponnorm, self.theta_data, self.dt, verbose=self.verbose
        # )
        # sol_notable = sdist_notable.fit()
        sdist_table = sdists.SlidingDist(
            sp_dists.exponnorm,
            self.theta_data,
            self.dt,
            tabulate_cdf=True,
            verbose=self.verbose,
        )
        sol_table = sdist_table.fit()
        qq_table = sdist_table.cdf(sol_table)
        data_table = sdist_table.ppf(sol_table, quantiles=qq_table)
        npt.assert_almost_equal(self.theta_data, data_table)
        # qq_notable = sdist_notable.cdf(sol_notable)
        # npt.assert_almost_equal(qq_notable, qq_table, decimal=5)


if __name__ == "__main__":
    npt.run_module_suite()
