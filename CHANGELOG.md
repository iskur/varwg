# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.7] - 2026-02-01

### Added
- GitLab CI/CD pipeline configuration for additional CI/CD integration

### Fixed
- Cross-process file-based locking for shelve database access to prevent race conditions during concurrent simulations
- Handle relative paths in `shelve_open()` without directory component
- Exclude test data files (test_data/*.nc) from published wheels to fix ARM64 validation errors and reduce wheel size

### Infrastructure
- PyPI publication now passes validation for all platforms including macOS ARM64

## [1.4.6] - 2026-01-29

### Added
- Detailed logging for wheel repacking issues for CI debugging

### Fixed
- Add missing `pytz` transitive dependency (required by timezonefinder)
- Exclude Cython-generated .c/.h files from binary wheels to resolve PyPI validation errors
- Improve macOS wheel repacking to eliminate trailing ZIP data
- Remove unnecessary zipfile-deflate64 dependency

### Changed
- Update macOS wheel build workflow for better delocate artifact handling and reproducibility

## [1.4.5] - 2026-01-23

### Added
- `obs_title_str` and `sim_title_str` parameters to `plot_meteogram_daily()` for customizable plot titles

## [1.4.4] - 2026-01-13

### Fixed
- NumPy 2.4+ compatibility: Replaced deprecated `np.sum(generator)` with builtin `sum()` in spectral.py

### Changed
- CI now uses `uv sync` to respect dependency lockfile for reproducible builds
- Updated uv.lock to match pyproject.toml version

### Documentation
- Added CLAUDE.md with development guidance and environment setup
- Regenerated README plots

## [1.4.3] - 2024-11-20

### Fixed
- Import `kron` from `numpy` instead of `scipy.linalg`

### Documentation
- Regenerated README plots

## [1.4.2] - 2025-11-18

### Added
- Thread-local random number generator via `varwg.get_rng()` for thread-safe concurrent simulations
- Thread-safe file locking for shelve cache operations
- Comprehensive thread-safety tests in `test_thread_local_rng.py`

### Changed
- All internal RNG usage migrated to thread-local `get_rng()` function
- `varwg.reseed()` now operates on thread-local RNG (only affects current thread)
- Updated Cython code in `cresample.pyx` to use thread-local RNG

### Deprecated
- Direct use of `varwg.rng` is deprecated and not thread-safe; use `varwg.get_rng()` instead

### Fixed
- Thread-safety issues in concurrent simulation scenarios
- Shelve cache race conditions with file-level locking
- Import error in plotting module (fixed `varwg.times` references)
- Mean-shifting bug in Z-transform for weathercop compatibility

## [1.4.1] - 2025-11-07

### Added
- Config validation in `set_conf()`

### Changed
- Output type specifications in `np.vectorize` calls
- Renamed VG to VarWG in test fixtures
- Removed `src/vg/test_data` directory

### Fixed
- RECORD file corruption in macOS wheels

### Infrastructure
- Updated macOS versions in CI workflow

## [1.4.0] - 2025-10-17

### Added
- Ridge regularization with unbiased parameter estimation option for better model stability
- Configurable SVAR parameters (FFT order and day-of-year width)
- Quantile-quantile shifting (`qq_shift`) for targeted climate scenarios with non-Gaussian variables
- Seasonal median support in distribution classes
- ECDF (empirical cumulative distribution function) exposed in core API
- Heat index calculations in meteox2y module
- Brunner compound index module with Cython optimization
- Automatic timezone offset calculation from latitude/longitude
- `to_df()` method with optional unit conversions from config
- Docker support for simulation deployment
- Tox integration for testing against installed wheels
- Python 3.14 wheel support
- Improved VG object serialization and `__str__` representation

### Changed
- Extracted dwd_opendata as separate package (github.com/iskur/dwd_opendata)
- Migrated test suite from nosetests to pytest
- Enhanced time parsing logic
- Improved KDE with performance caching
- Better NaN/non-finite value handling across codebase
- Applied black code formatting consistently (line-length: 79)

### Fixed
- Circular import issues with times/ctimes modules
- Matrix inversion stability improvements
- Distribution fitting mode extraction and NaN handling
- Disaggregation bounds handling and seasonal cycles
- Plotting contour level generation and canvas title issues

### Removed
- Python 2 compatibility code
- Monty simulation scripts

### Infrastructure
- Multi-platform wheel distribution via cibuildwheel
- Pre-built wheels for Python 3.13 on Linux (x86_64), Windows (AMD64), macOS (x86_64, arm64)
- Installation no longer requires C compiler on supported platforms
- Added `.github/workflows/build-wheels.yml` for automated multi-platform builds

## [1.3.0]

### Added
- Missing value infilling using VAR processes prior to simulation
- Phase-randomization of VAR-residuals for better low-frequency variability reproduction
- WeatherCop orchestration capability for multi-site data generation

### Changed
- Migrated to modern build system with pyproject.toml
- Dependency management with uv
- Moved to src/vg/ package layout

### Requirements
- Python ≥ 3.13 required

## [1.2.0]

### Added
- Scenarios guided through changes in non-normally distributed variables
- Disaggregation recreates seasonal changes in daily cycles
- Support for all scipy.stats.distributions for variable fitting

### Fixed
- Disaggregation bugs in presence of NaNs

## [1.1.0]

### Added
- Support for non-evenly spaced time series (uses linear interpolation)
- Support for gaps/NaNs (ignored by estimators, not interpolated)
- Disaggregation works on variables with bounds

## [1.0.0]

Initial release.
