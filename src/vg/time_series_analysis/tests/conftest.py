"""Shared pytest fixtures for VG time_series_analysis tests."""
import pytest
from pathlib import Path
import xarray as xr


@pytest.fixture
def fixture_dir():
    """Path to test fixture data directory."""
    return Path(__file__).parent.parent.parent / "test_data"


@pytest.fixture
def stoetten_precipitation(fixture_dir):
    """Load Stötten precipitation fixture data."""
    fixture_path = fixture_dir / "stoetten_precipitation_daily_1990_2020.nc"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    return xr.open_dataarray(fixture_path)


@pytest.fixture
def konstanz_temperature(fixture_dir):
    """Load Konstanz air temperature fixture data."""
    fixture_path = fixture_dir / "konstanz_temperature_daily_2000_2016.nc"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    return xr.open_dataarray(fixture_path)


@pytest.fixture
def konstanz_sun(fixture_dir):
    """Load Konstanz sun fixture data."""
    fixture_path = fixture_dir / "konstanz_sun_daily_2000_2016.nc"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    return xr.open_dataarray(fixture_path)


@pytest.fixture
def freiburg_precipitation(fixture_dir):
    """Load Freiburg precipitation fixture data."""
    fixture_path = fixture_dir / "freiburg_precipitation_daily_2000_2016.nc"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    return xr.open_dataarray(fixture_path)
