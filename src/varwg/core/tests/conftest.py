"""Pytest fixtures for VarWG tests."""

from pathlib import Path
import tempfile

import numpy as np
import pytest

import varwg

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
script_home = Path(varwg.__file__).parent
met_file = script_home / "sample.met"
data_dir = Path(varwg.core.__file__).parent / "tests" / "data"
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
    met = varwg.read_met(sim_file, verbose=False, with_conversions=True)[1]
    return np.array([met[var_name] for var_name in var_names])


@pytest.fixture(scope="session")
def varwg_kwds(test_data_dir):
    """Common VarWG initialization keywords."""
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
def varwg_regr(varwg_kwds):
    """Session-scoped VarWG instance with regression rain method."""
    varwg.reseed(seed)
    met_varwg = varwg.VarWG(var_names, rain_method="regression", **varwg_kwds)
    met_varwg.fit(**fit_kwds)
    return met_varwg


@pytest.fixture(scope="session")
def varwg_dist(varwg_kwds):
    """Session-scoped VarWG instance with distance rain method."""
    met_varwg = varwg.VarWG(var_names, rain_method="distance", **varwg_kwds)
    met_varwg.fit(**fit_kwds)
    return met_varwg


@pytest.fixture(scope="session")
def varwg_sim(varwg_kwds):
    """Session-scoped VarWG instance with simulation rain method."""
    met_varwg = varwg.VarWG(var_names, rain_method="simulation", **varwg_kwds)
    met_varwg.fit(**fit_kwds)
    return met_varwg


@pytest.fixture
def varwg_regr_fresh(varwg_kwds):
    """Function-scoped VarWG instance for tests that need to call fit() with different params."""
    varwg.reseed(seed)
    met_varwg = varwg.VarWG(var_names, rain_method="regression", **varwg_kwds)
    met_varwg.fit(**fit_kwds)
    return met_varwg
