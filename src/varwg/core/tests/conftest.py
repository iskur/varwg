"""Pytest fixtures for VG tests."""

from pathlib import Path
import tempfile

import numpy as np
import pytest

import vg

# Test configuration
seed = 0
p = 3
fit_kwds = dict(p=p, fft_order=3, doy_width=15, seasonal=True)
var_names = (
    "R",
    "theta",
    "Qsw",
    "ILWR",
    "rh",
    "u",
    "v",
)

# File paths
script_home = Path(vg.__file__).parent
met_file = script_home / "sample.met"
data_dir = Path(vg.core.__file__).parent / "tests" / "data"
sim_file = data_dir / "test_out_sample.met"


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-wide temporary directory for test data."""
    tmpdir = tempfile.mkdtemp("vg_test_session")
    yield tmpdir
    # Note: Not cleaning up for now to allow inspection
    # import shutil
    # shutil.rmtree(tmpdir)


@pytest.fixture(scope="session")
def sample_sim():
    """Load sample simulation data once per session."""
    met = vg.read_met(sim_file, verbose=False, with_conversions=True)[1]
    return np.array([met[var_name] for var_name in var_names])


@pytest.fixture(scope="session")
def vg_kwds(test_data_dir):
    """Common VG initialization keywords."""
    return dict(
        refit=True,
        data_dir=test_data_dir,
        cache_dir=test_data_dir,
        met_file=met_file,
        verbose=False,
        infill=True,
        station_name="test",
    )


@pytest.fixture(scope="session")
def vg_regr(vg_kwds):
    """Session-scoped VG instance with regression rain method."""
    vg.reseed(seed)
    met_vg = vg.VG(var_names, rain_method="regression", **vg_kwds)
    met_vg.fit(**fit_kwds)
    return met_vg


@pytest.fixture(scope="session")
def vg_dist(vg_kwds):
    """Session-scoped VG instance with distance rain method."""
    met_vg = vg.VG(var_names, rain_method="distance", **vg_kwds)
    met_vg.fit(**fit_kwds)
    return met_vg


@pytest.fixture(scope="session")
def vg_sim(vg_kwds):
    """Session-scoped VG instance with simulation rain method."""
    met_vg = vg.VG(var_names, rain_method="simulation", **vg_kwds)
    met_vg.fit(**fit_kwds)
    return met_vg


@pytest.fixture
def vg_regr_fresh(vg_kwds):
    """Function-scoped VG instance for tests that need to call fit() with different params."""
    vg.reseed(seed)
    met_vg = vg.VG(var_names, rain_method="regression", **vg_kwds)
    met_vg.fit(**fit_kwds)
    return met_vg
