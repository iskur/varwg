# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2025-10-14

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
