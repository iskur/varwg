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
