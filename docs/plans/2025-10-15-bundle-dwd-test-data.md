# Bundle DWD Test Data Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Bundle necessary DWD weather data to make CI tests pass without network access.

**Architecture:** Create fixture data from actual DWD downloads, bundle it with the package, modify tests to use bundled data when available, keep network tests for optional validation.

**Tech Stack:** pytest fixtures, package_data, pytest marks, xarray/pandas

---

## Task 1: Analyze Test Data Requirements

**Files:**
- Read: `src/vg/tests/test_brunner.py:50-67`
- Read: `src/vg/tests/test_meteox2y.py:49-65`
- Read: `src/vg/time_series_analysis/tests/test_seasonal_distributions.py:41-267`

**Step 1: Identify required test data**

Document which stations, variables, and time ranges are needed:
- test_brunner.py: Stötten station, precipitation, hourly, 1990-2020
- test_meteox2y.py: Stötten station, precipitation, hourly, 1990-2020
- test_seasonal_distributions.py: Konstanz (air_temperature, sun), Freiburg (precipitation), 2000-2016

**Step 2: Create data requirements document**

Create a temporary notes file documenting:
```
# Test Data Requirements

## test_brunner.py::test_SPI
- Station: Stötten
- Variable: precipitation
- Time: hourly
- Range: 1990-2020

## test_meteox2y.py::test_SPI
- Station: Stötten
- Variable: precipitation
- Time: hourly
- Range: 1990-2020

## test_seasonal_distributions.py::Test
- Station: Konstanz, air_temperature, 2000-2016
- Station: Konstanz, sun, 2000-2016
- Station: Freiburg, precipitation, 2000-2016
```

**Step 3: Verify current test behavior**

Run: `pytest src/vg/tests/test_brunner.py::Test::test_SPI -v`
Expected: May pass if network available, or be skipped

---

## Task 2: Download and Bundle Test Data

**Files:**
- Create: `src/vg/test_data/`
- Create: `src/vg/test_data/__init__.py`
- Create: `src/vg/test_data/download_fixtures.py`

**Step 1: Write the fixture download script**

```python
"""Download DWD test fixtures for offline testing."""
from pathlib import Path
import xarray as xr
from vg.meteo import dwd_opendata

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
```

**Step 2: Run the download script**

Run: `cd src/vg/test_data && python download_fixtures.py`
Expected: Creates 4 .nc files in src/vg/test_data/

**Step 3: Create test_data package init**

```python
"""Test fixtures for VG package tests."""
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent

def get_fixture_path(filename):
    """Get path to a test fixture file."""
    return FIXTURE_DIR / filename
```

**Step 4: Verify fixture files exist**

Run: `ls -lh src/vg/test_data/*.nc`
Expected: Lists 4 netCDF files with reasonable sizes

**Step 5: Commit test data**

```bash
git add src/vg/test_data/
git commit -m "feat: add bundled DWD test fixtures for offline testing"
```

---

## Task 3: Update pyproject.toml Package Data

**Files:**
- Modify: `pyproject.toml:32-33`

**Step 1: Add test data to package_data**

```toml
[tool.setuptools.package-data]
vg = ["*.met", "test_data/*.nc"]
```

**Step 2: Verify package data will be included**

Run: `uv build && unzip -l dist/vg-*.whl | grep test_data`
Expected: Shows test_data/*.nc files in the wheel

**Step 3: Commit configuration change**

```bash
git add pyproject.toml
git commit -m "build: include test_data fixtures in package"
```

---

## Task 4: Create Fixture Loading Utilities

**Files:**
- Create: `src/vg/tests/conftest.py`

**Step 1: Write pytest fixtures for test data**

```python
"""Shared pytest fixtures for VG tests."""
import pytest
from pathlib import Path
import xarray as xr


@pytest.fixture
def fixture_dir():
    """Path to test fixture data directory."""
    return Path(__file__).parent.parent / "test_data"


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
```

**Step 2: Test fixture loading**

Run: `pytest --fixtures src/vg/tests/ | grep -A2 "stoetten_precipitation"`
Expected: Shows the fixture with its docstring

**Step 3: Commit conftest**

```bash
git add src/vg/tests/conftest.py
git commit -m "test: add pytest fixtures for bundled test data"
```

---

## Task 5: Update test_brunner.py

**Files:**
- Modify: `src/vg/tests/test_brunner.py:50-67`

**Step 1: Remove network marker and use fixture**

```python
def test_SPI(self, stoetten_precipitation):
    prec = stoetten_precipitation
    spi = brunner.SPI_ar(prec, weeks=6)
```

**Step 2: Add network test variant**

```python
@pytest.mark.network
def test_SPI_network(self):
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
```

**Step 3: Run the updated test**

Run: `pytest src/vg/tests/test_brunner.py::Test::test_SPI -v`
Expected: PASS (using fixture, no network needed)

**Step 4: Run network test to verify it still works**

Run: `pytest src/vg/tests/test_brunner.py::Test::test_SPI_network -v -m network`
Expected: PASS (if network available) or SKIPPED

**Step 5: Commit test changes**

```bash
git add src/vg/tests/test_brunner.py
git commit -m "test: use bundled fixtures for test_SPI, add network variant"
```

---

## Task 6: Update test_meteox2y.py

**Files:**
- Modify: `src/vg/tests/test_meteox2y.py:49-65`

**Step 1: Add fixture parameter to test**

```python
def test_SPI(self, stoetten_precipitation):
    prec = stoetten_precipitation
    spi = meteox2y.SPI_ar(prec, weeks=6)
```

**Step 2: Add network test variant**

```python
@pytest.mark.network
def test_SPI_network(self):
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
    spi = meteox2y.SPI_ar(prec, weeks=6)
```

**Step 3: Run the updated test**

Run: `pytest src/vg/tests/test_meteox2y.py::Test::test_SPI -v`
Expected: PASS (using fixture)

**Step 4: Commit test changes**

```bash
git add src/vg/tests/test_meteox2y.py
git commit -m "test: use bundled fixtures for meteox2y test_SPI"
```

---

## Task 7: Update test_seasonal_distributions.py

**Files:**
- Modify: `src/vg/time_series_analysis/tests/test_seasonal_distributions.py:41-267`

**Step 1: Refactor Test class to use fixtures in setUp**

```python
class Test(npt.TestCase):
    def setUp(self, konstanz_temperature, konstanz_sun):
        self.verbose = True
        self.cache_dir = Path(tempfile.mkdtemp("vg_test"))

        # Use fixture data instead of downloading
        theta_xr = konstanz_temperature
        self.theta_data = theta_xr.values.squeeze()
        self.dt = theta_xr.time.to_dataframe().index.to_pydatetime()
```

Wait, TestCase classes don't work well with fixtures. Let me revise:

**Step 1: Convert test class to use pytest fixtures properly**

Remove the `@pytest.mark.network` decorator from the class and add it to individual methods that need network. Create new fixture-based tests:

```python
@pytest.fixture
def theta_setup(konstanz_temperature):
    """Setup temperature data for tests."""
    theta_xr = konstanz_temperature
    theta_data = theta_xr.values.squeeze()
    dt = theta_xr.time.to_dataframe().index.to_pydatetime()
    return theta_data, dt


def test_cdf_table(theta_setup):
    """Test CDF tabulation with fixture data."""
    theta_data, dt = theta_setup
    verbose = True

    sdist_table = sdists.SlidingDist(
        sp_dists.exponnorm,
        theta_data,
        dt,
        tabulate_cdf=True,
        verbose=verbose,
    )
    sol_table = sdist_table.fit()
    qq_table = sdist_table.cdf(sol_table)
    data_table = sdist_table.ppf(sol_table, quantiles=qq_table)
    npt.assert_almost_equal(theta_data, data_table)


def test_serialization(theta_setup, tmp_path):
    """Test serialization with fixture data."""
    theta_data, dt = theta_setup
    verbose = True

    sdist_orig = sdists.SlidingDist(
        sp_dists.exponnorm, theta_data, dt, verbose=verbose
    )
    sh = shelve.open(str(tmp_path / "seasonal_cache_file"), "c")
    sh["theta"] = sdist_orig
    sh.close()
    sh = shelve.open(str(tmp_path / "seasonal_cache_file"), "c")
    sdist_shelve = sh["theta"]
    assert not my.recursive_diff(
        None,
        sdist_orig,
        sdist_shelve,
        verbose=True,
        plot=True,
        ignore_types=(sp_dists.rv_continuous,),
    )


def test_rainmix(freiburg_precipitation):
    """Test rainmix with fixture data."""
    prec_xr = freiburg_precipitation
    prec_data = np.squeeze(prec_xr.values)
    dt = prec_xr.time.to_dataframe().index.to_pydatetime()
    threshold = 0.001 * 24
    verbose = True

    dist = dists.RainMix(
        dists.kumaraswamy,
        threshold=threshold,
    )
    fixed_pars = dict(
        u=(lambda x: np.ones_like(x)), l=(lambda x: np.zeros_like(x))
    )
    sdist = sdists.SlidingDist(
        dist,
        prec_data,
        dt,
        doy_width=15,
        verbose=verbose,
        fixed_pars=fixed_pars,
        tabulate_cdf=True,
    )
    sol = sdist.fit()
    qq = sdist.cdf(sol)
    prec_recovered = sdist.ppf(sol, qq)
    assert all(prec_recovered >= 0)
    rain_mask = prec_data > threshold
    npt.assert_allclose(
        prec_recovered[rain_mask],
        prec_data[rain_mask],
    )


def test_rainmix_sun(konstanz_sun):
    """Test rainmix sun with fixture data."""
    sun_xr = konstanz_sun
    sun_data = np.squeeze(sun_xr.values)
    dt = sun_xr.time.to_dataframe().index.to_pydatetime()

    longitude = 47.66
    latitude = 9.18

    def max_sunshine_hours(doys):
        from vg import times
        dates = times.doy2datetime(doys)
        sun_hours = meteox2y.sunshine_hours(
            dates,
            longitude=longitude,
            latitude=latitude,
            tz_offset=0,
        )
        return sun_hours * 60

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
        doy_width=15,
        verbose=True,
        fixed_pars=fixed_pars,
        tabulate_cdf=True,
    )
    sol = sdist.fit()
    qq = sdist.cdf(sol)
    sun_back = sdist.ppf(sol, qq)
    over_thresh = sun_back > sdist.dist.thresh
    npt.assert_allclose(
        sun_back[over_thresh], sun_data[over_thresh], atol=0.0031
    )
```

**Step 2: Keep network test class for integration testing**

```python
@pytest.mark.network
class TestWithNetwork(npt.TestCase):
    """Network-dependent integration tests that download fresh data."""

    def setUp(self):
        self.verbose = True
        self.cache_dir = Path(tempfile.mkdtemp("vg_test"))

        theta_xr = dwd_opendata.load_station(
            "Konstanz", "air_temperature"
        ).squeeze()
        # ... rest of original setUp code

    # ... all original test methods
```

**Step 3: Run fixture-based tests**

Run: `pytest src/vg/time_series_analysis/tests/test_seasonal_distributions.py -v -m "not network"`
Expected: All new fixture-based tests PASS

**Step 4: Run network tests**

Run: `pytest src/vg/time_series_analysis/tests/test_seasonal_distributions.py -v -m network`
Expected: Original network tests still work

**Step 5: Commit test refactoring**

```bash
git add src/vg/time_series_analysis/tests/test_seasonal_distributions.py
git commit -m "test: refactor seasonal_distributions to use bundled fixtures"
```

---

## Task 8: Verify CI Success

**Files:**
- Read: `.github/workflows/ci.yml:37`

**Step 1: Run local CI simulation**

Run: `pytest --pyargs vg -m "not network" -v`
Expected: All tests pass without network access

**Step 2: Check for any remaining network-dependent tests**

Run: `pytest --pyargs vg -v --collect-only | grep -i network`
Expected: Shows only tests explicitly marked with @pytest.mark.network

**Step 3: Test the build and install workflow**

```bash
uv venv --python 3.13
source .venv/bin/activate
uv build
uv pip install dist/*.whl
pytest --pyargs vg -m "not network" -v
```

Expected: All tests pass with installed wheel

**Step 4: Push and verify CI**

```bash
git push origin HEAD
```

Expected: CI workflow passes on GitHub Actions

**Step 5: Document the change**

Create docs/plans/test-data-bundling-complete.md with summary:
- What data was bundled
- How to regenerate fixtures (run download_fixtures.py)
- How network tests still work with -m network flag

---

## Task 9: Add Documentation

**Files:**
- Create: `src/vg/test_data/README.md`

**Step 1: Document the test data system**

```markdown
# VG Test Data Fixtures

This directory contains bundled weather data from DWD (Deutscher Wetterdienst)
for offline testing.

## Fixture Files

- `stoetten_precipitation_daily_1990_2020.nc`: Stötten station, daily precipitation, 1990-2020
- `konstanz_temperature_daily_2000_2016.nc`: Konstanz station, daily air temperature, 2000-2016
- `konstanz_sun_daily_2000_2016.nc`: Konstanz station, daily sunshine, 2000-2016
- `freiburg_precipitation_daily_2000_2016.nc`: Freiburg station, daily precipitation, 2000-2016

## Regenerating Fixtures

To regenerate test fixtures from fresh DWD downloads:

```bash
cd src/vg/test_data
python download_fixtures.py
```

Requires network access to DWD OpenData FTP servers.

## Using Fixtures in Tests

Fixtures are automatically available via pytest conftest.py:

```python
def test_something(stoetten_precipitation):
    # Use the fixture data
    assert stoetten_precipitation is not None
```

## Network Tests

Tests that download fresh data should be marked with `@pytest.mark.network`:

```python
@pytest.mark.network
def test_with_live_data():
    data = dwd_opendata.load_station("Konstanz", "air_temperature")
    # ...
```

Run network tests with: `pytest -m network`
Skip network tests with: `pytest -m "not network"` (CI default)
```

**Step 2: Commit documentation**

```bash
git add src/vg/test_data/README.md
git commit -m "docs: add test data fixtures documentation"
```

---

## Task 10: Final Verification and Cleanup

**Files:**
- Verify: All modified test files

**Step 1: Run full test suite without network**

Run: `pytest --pyargs vg -m "not network" -v --tb=short`
Expected: All tests pass, no network access attempted

**Step 2: Run full test suite with network**

Run: `pytest --pyargs vg -v --tb=short`
Expected: All tests pass including network tests

**Step 3: Verify wheel contents**

Run: `unzip -l dist/vg-*.whl | grep -E "(test_data|\.nc)"`
Expected: Shows all 4 .nc fixture files included

**Step 4: Clean up any temporary files**

Run: `git status`
Expected: Only tracked files, no untracked artifacts

**Step 5: Final commit and summary**

```bash
git log --oneline -10
```

Expected: Shows clean commit history of the implementation

Create summary in commit message or PR description:
- Bundled 4 DWD weather station datasets as test fixtures
- Modified 3 test files to use fixtures by default
- Preserved network tests with explicit markers
- CI now passes without network access
- Network tests still available with `pytest -m network`
