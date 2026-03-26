"""Shared pytest fixtures for VG tests."""

import pytest
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from pytest_examples import find_examples


def find_project_root(marker_files=(".git", "pyproject.toml", "setup.py")):
    """Find project root by looking for marker files.

    Returns None if running from an installed package (e.g., via pytest --pyargs),
    where source-tree markers don't exist.
    """
    current = Path(__file__).resolve().parent

    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in marker_files):
            return parent

    return None


PROJECT_ROOT = find_project_root()


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


@pytest.fixture
def air_temperature_tutorial(fixture_dir):
    """Load air temperature tutorial dataset fixture (subset of xr.tutorial.load_dataset)."""
    fixture_path = fixture_dir / "air_temperature_tutorial_subset.nc"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    return xr.open_dataset(fixture_path)


@pytest.fixture
def plot_output_dir():
    """Path to save generated plots during example execution.

    When running from installed package (--pyargs), uses temp directory.
    """
    if PROJECT_ROOT is None:
        import tempfile
        return Path(tempfile.gettempdir()) / "varwg_plots"

    output_dir = PROJECT_ROOT / "docs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def save_example_plots(plot_output_dir):
    """Auto-save matplotlib figures after example execution."""
    yield

    # Save all open figures
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        # Try to save with a descriptive name if available
        filename = (
            getattr(fig, "name", False) or fig.get_label() or f"figure_{i}.png"
        )
        if not filename.endswith(".png"):
            filename = f"{filename}.png"
        fig.savefig(plot_output_dir / filename, dpi=150, bbox_inches="tight")

    plt.close("all")


def pytest_generate_tests(metafunc):
    """Generate parametrized tests for README examples.

    Skips when running from installed package (--pyargs) where README.md
    is not available.
    """
    if "readme_example" in metafunc.fixturenames:
        if PROJECT_ROOT is None:
            pytest.skip(
                "README examples require source tree (not available in --pyargs mode)"
            )
        # Type guard: PROJECT_ROOT is not None if we reach here
        readme_path: Path = PROJECT_ROOT / "README.md"
        examples = list(find_examples(readme_path))
        metafunc.parametrize("readme_example", examples, ids=str)
