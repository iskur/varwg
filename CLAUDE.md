# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VARWG (Vector-Autoregressive Weather Generator) is a single-site weather generator for hydrodynamic and ecologic lake modeling. It generates synthetic meteorological time series while preserving temporal correlations and seasonal patterns using Vector Autoregressive (VAR) models.

**Key capabilities:**
- Simulates multivariate meteorological data (temperature, radiation, humidity, wind, precipitation)
- Preserves correlations between variables and temporal dependencies
- Supports climate change scenario modeling (changes in mean, variability, correlations)
- Thread-safe concurrent simulations via thread-local RNG

**Technology stack:**
- Python 3.13+ (setuptools-based build with Cython extensions)
- uv for dependency management
- Cython for performance-critical resampling and time series operations
- NumPy/SciPy for numerical computations
- Pandas/xarray for time series data
- Cartopy/matplotlib for plotting and visualization

## Development Setup

**Install dependencies and build:**
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
uv sync --all-groups

# Build Cython extensions
uv pip install cython numpy setuptools
uv build
uv pip install dist/*.whl
```

**Running tests:**
```bash
# Run all tests (requires network for some tests)
uv run pytest

# Skip network-dependent tests
uv run pytest -m "not network"

# Run specific test module
uv run pytest src/varwg/tests/test_brunner.py

# Run tests against installed wheel (validates packaging)
uv run tox
```

**Formatting:**
Code is auto-formatted with `ruff format` (via PostToolUse hook in this environment). Line length: 79 characters.

## Architecture

### Core Components

**`varwg.VarWG` (main class in `core/core.py`):**
The primary interface for weather generation. Inherits from `VGBase` (data handling, model fitting) and `VGPlotting` (visualization).

Key workflow:
1. Initialize with variable names and meteorological data file
2. Fit seasonal VAR model with `fit(p=3, seasonal=True)`
3. Simulate synthetic data with `simulate(T=days)`
4. Optionally apply climate scenarios during simulation

**Base classes (`core/base.py`):**
- `VGBase`: Core data handling, model parameter estimation, caching
- `VGPlotting` (`core/plotting.py`): Plotting utilities (meteograms, seasonal cycles)

**Configuration system (`config_template.py`):**
Global configuration via `varwg.set_conf(config_module)` sets:
- Geographic coordinates (for solar radiation calculations)
- Data directories and cache locations
- Distribution specifications for variables
- Variable name mappings and output formats

Configuration is propagated to all submodules: `core.conf`, `base.conf`, `plotting.conf`.

### Time Series Analysis Module (`time_series_analysis/`)

Statistical machinery for VAR model fitting and simulation:

- **`models.py`**: VAR/VARX parameter estimation (least squares, ridge regularization), model selection criteria (AIC, BIC, HQ), MGARCH support
- **`seasonal_distributions.py`/`seasonal_kde.py`**: Seasonal probability distributions using kernel density estimation for transforming residuals
- **`distributions.py`**: Marginal distribution transformations, empirical CDFs
- **`resample.py`**: Data aggregation and temporal resampling
- **`cresample.pyx`**: Cython-accelerated resampling (critical for performance)
- **`time_series.py`**: Utility functions for time series operations
- **`phase_randomization.py`**: Spectral methods for preserving power spectra

**VAR Model Architecture:**
- Seasonal VAR models split year into periods (default: 73 5-day periods)
- Each period has its own VAR parameters, distributions, and correlation structure
- Transitions between periods use interpolation to avoid discontinuities
- Residuals transformed via seasonal KDE to match observed distributions

### Meteorological Utilities (`meteo/`)

- **`meteox2y.py`**: Conversions between meteorological variables (e.g., RH → dewpoint, compute potential solar radiation)
- **`meteox2y_cy.pyx`**: Cython versions for performance-critical conversions
- **`avrwind.py`**: Wind direction/speed handling (components ↔ angles)
- **`brunner.py`**: Precipitation modeling

### Cython Extensions

Three performance-critical modules compiled to C:
- **`time_series_analysis/cresample.pyx`**: Fast temporal resampling with OpenMP (Linux only)
- **`meteo/meteox2y_cy.pyx`**: Fast meteorological computations with OpenMP
- **`ctimes.pyx`**: Date/time handling utilities

Extensions are built via `setup.py` which handles platform-specific compilation flags (OpenMP on Linux, none on macOS/Windows).

## Data Flow

1. **Input**: Meteorological data file (`.met` format or custom via `read_met()`)
   - Tab-delimited with flexible date/time parsing
   - Typically hourly observations aggregated to daily

2. **Preprocessing**:
   - Parse timestamps via `_parse_time()` (supports multiple formats)
   - Aggregate to daily values via `sumup()` or `met_as_array()`
   - Apply variable transformations (e.g., wind components)

3. **Model Fitting**:
   - Split data into seasonal periods
   - Fit VAR(p) model per period via least squares
   - Estimate residual distributions via kernel density estimation
   - Cache fitted parameters in shelve database (thread-safe)

4. **Simulation**:
   - Draw VAR innovations from historical residual distributions
   - Apply autoregressive dynamics: `Y_t = c + A₁Y_{t-1} + ... + AₚY_{t-p} + ε_t`
   - Transform via inverse seasonal distributions to match marginals
   - Apply optional climate scenarios (delta changes, variance scaling)

5. **Output**:
   - Simulated time series as NumPy arrays or pandas DataFrames
   - Visualization via plotting methods
   - Export to ASCII via `dump_data()`

## Testing Strategy

- **Unit tests**: Individual functions in `tests/` subdirectories
- **Integration tests**: Full workflows in `src/varwg/tests/`
- **Network tests**: Marked with `@pytest.mark.network` (require internet for downloading test data)
- **Fixture data**: Cached in `src/varwg/test_data/` or downloaded via `pooch`

**Test isolation**: tox builds wheel and installs in clean virtualenv to catch import issues and circular dependencies.

## Thread Safety

**Random number generation**: Thread-local RNG via `varwg.get_rng()`. Each thread gets independent Generator instance. Do NOT use deprecated `varwg.rng` directly.

**Cache access**: Shelve database operations protected by threading.Lock to prevent race conditions during concurrent simulations.

## Common Pitfalls

1. **Configuration not set**: Must call `varwg.set_conf()` before using VarWG. Template available as `varwg.config_template`.

2. **Cython rebuild**: After editing `.pyx` files, rebuild extensions:
   ```bash
   uv build && uv pip install --force-reinstall dist/*.whl
   ```

3. **Import order**: Cython modules must be compiled before importing. If seeing "No module named 'varwg.ctimes'", rebuild package.

4. **Platform differences**: OpenMP only enabled on Linux. macOS/Windows use single-threaded Cython extensions.

5. **Backward compatibility**: Old class names (`VG`, `VGBase`, `VGPlotting`) are aliased to new names (`VarWG`, etc.) in `__init__.py`.

6. **Deprecation warnings**: Currently have NumPy/SciPy deprecation warnings to address:
   - `core/core.py:237`: NumPy scalar conversion deprecation (ensure single element before `float()` conversion)
   - `helpers.py:327`: SciPy import path deprecation (import `interp1d` from `scipy.interpolate` directly)
   - These are non-blocking but should be fixed in a future maintenance release to maintain compatibility with NumPy 2.0+ and SciPy 2.0+

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Build wheels via GitHub Actions (`.github/workflows/build-wheels.yml`)
4. Wheels built for: Linux x86_64, Windows AMD64, macOS x86_64/arm64
5. Test wheel installation with `tox`
6. Upload to PyPI via `twine`

Pre-built wheels include compiled Cython extensions, so end users don't need a C compiler.