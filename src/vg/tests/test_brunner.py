import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from vg.meteo import meteox2y as brunner
from vg.meteo import dwd_opendata


class Test(npt.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_brunner_compound(self):
        Ta = np.arange(10)
        P = Ta[::-1]
        hot_dry = brunner.brunner_compound(Ta, P)
        npt.assert_almost_equal(
            hot_dry, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        hot_dry_seq = brunner.brunner_compound(Ta, P, sequential=True)
        npt.assert_almost_equal(hot_dry_seq, hot_dry)

        rs = np.random.RandomState(0)
        Ta = xr.DataArray(
            rs.randn(2, 100, 1000), dims=["model", "station", "time"]
        )
        P = Ta + 0.5 * rs.randn(*Ta.shape)
        Ta.data[0, 0, 0] = np.nan
        P.data[0, 0, 1] = np.nan
        bc_full = brunner.brunner_compound(Ta, P)
        assert np.all(np.isnan(bc_full[0, 0, :2]))
        assert np.all(np.isfinite(bc_full[0, 0, 2:]))
        assert bc_full.shape == Ta.shape
        assert np.nanmax(bc_full) <= 1
        assert np.nanmin(bc_full) >= 0
        bc_seq = brunner.brunner_compound(Ta, P, sequential=True)
        npt.assert_almost_equal(bc_seq, bc_full, decimal=4)

    def test_STI(self):
        temperature = xr.tutorial.load_dataset("air_temperature")
        sti = brunner.STI_ar(temperature["air"].isel(lat=10, lon=40), weeks=3)
        sti_ds = brunner.STI_ds(
            temperature.isel(lat=slice(10, 12), lon=slice(39, 42)), weeks=3
        )

    @pytest.mark.network
    def test_SPI(self):
        """Test SPI with live DWD download (network required)."""
        prec = dwd_opendata.load_station(
            "Stötten", "precipitation", time="hourly"
        )
        prec = (
            prec.interpolate_na("time")
            .squeeze()
            .resample(time="1d")
            .sum()
            .sel(time=slice("1990", "2020"))
            .dropna("time")
        )
        spi = brunner.SPI_ar(prec, weeks=6)


# Fixture-based tests (use bundled data, no network required)
def test_SPI_fixture(stoetten_precipitation):
    """Test SPI with bundled fixture data (no network required)."""
    prec = stoetten_precipitation
    spi = brunner.SPI_ar(prec, weeks=6)
    # Basic validation
    assert spi is not None
    assert len(spi) == len(prec)
