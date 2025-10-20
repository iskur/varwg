"""Download DWD test fixtures for offline testing."""
from pathlib import Path
import xarray as xr
from varwg.meteo import dwd_opendata

FIXTURE_DIR = Path(__file__).parent
FIXTURE_DIR.mkdir(exist_ok=True)

def download_stoetten_precipitation():
    """Download Stötten precipitation data for test_brunner and test_meteox2y."""
    print("Downloading Stötten precipitation...")
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
    prec.to_netcdf(FIXTURE_DIR / "stoetten_precipitation_daily_1990_2020.nc")
    print(f"Saved to {FIXTURE_DIR / 'stoetten_precipitation_daily_1990_2020.nc'}")

def download_konstanz_temperature():
    """Download Konstanz air temperature for test_seasonal_distributions."""
    print("Downloading Konstanz air temperature...")
    theta_xr = dwd_opendata.load_station(
        "Konstanz", "air_temperature"
    ).squeeze()
    theta_xr = (
        theta_xr.sel(time=slice("2000", "2016"))
        .resample(time="D")
        .mean()
        .interpolate_na("time")
    )
    theta_xr.to_netcdf(FIXTURE_DIR / "konstanz_temperature_daily_2000_2016.nc")
    print(f"Saved to {FIXTURE_DIR / 'konstanz_temperature_daily_2000_2016.nc'}")

def download_konstanz_sun():
    """Download Konstanz sun data for test_seasonal_distributions."""
    print("Downloading Konstanz sun...")
    sun_xr = dwd_opendata.load_station("Konstanz", "sun").squeeze()
    sun_xr = (
        sun_xr.sel(time=slice("2000", "2016"))
        .resample(time="D")
        .mean()
        .interpolate_na("time")
    )
    sun_xr.to_netcdf(FIXTURE_DIR / "konstanz_sun_daily_2000_2016.nc")
    print(f"Saved to {FIXTURE_DIR / 'konstanz_sun_daily_2000_2016.nc'}")

def download_freiburg_precipitation():
    """Download Freiburg precipitation for test_seasonal_distributions."""
    print("Downloading Freiburg precipitation...")
    prec_xr = dwd_opendata.load_station("Freiburg", "precipitation")
    prec_xr = (
        prec_xr.sel(time=slice("2000", "2016"))
        .resample(time="D")
        .sum()
        .interpolate_na("time")
    )
    prec_xr.to_netcdf(FIXTURE_DIR / "freiburg_precipitation_daily_2000_2016.nc")
    print(f"Saved to {FIXTURE_DIR / 'freiburg_precipitation_daily_2000_2016.nc'}")

if __name__ == "__main__":
    download_stoetten_precipitation()
    download_konstanz_temperature()
    download_konstanz_sun()
    download_freiburg_precipitation()
    print("\nAll fixtures downloaded successfully!")
