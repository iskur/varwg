# VG: Vector Autoregressive Moving Average Weather generator

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/vg-doc/badge/?version=latest)](https://vg-doc.readthedocs.io)
[![Managed with uv](https://img.shields.io/badge/managed_with-uv-blue)](https://github.com/astral-sh/uv)
![Last Commit](https://img.shields.io/github/last-commit/iskur/vg)

## What is VG?

The weather generator VG is a single-site Vector-Autoregressive (Moving
Average) weather generator that was developed for hydrodynamic and
ecologic modelling of lakes. It includes a number of possibilities to
define climate scenarios. For example, changes in mean or in the
variability of air temperature can be set. Correlations during
simulations are preserved, so that these changes propagate from the air
temperature to the other simulated variables.

## Installation

### Using uv (recommended)

VG uses [uv](https://docs.astral.sh/uv/) for dependency management. To install:

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and install:
   ```bash
   git clone https://github.com/iskur/vg.git
   cd vg
   uv sync
   ```

### Using pip

Alternatively, you can install using pip:

```bash
pip install git+https://github.com/iskur/vg.git
```

### Development installation

For development with additional tools:

```bash
uv sync --group dev
```

## Quick Start

After installation, you can use VG to generate synthetic weather data:

```python
import vg
from pathlib import Path

# Configure VG with default settings
vg.set_conf(vg.config_template)

# Define meteorological variables to simulate
var_names = ("theta", "Qsw", "rh")  # Temperature, solar radiation, humidity

# Initialize the weather generator with sample data
met_vg = vg.VG(var_names, met_file=vg.sample_met, refit=True, verbose=True)

# Fit the seasonal VAR model
met_vg.fit(p=3, seasonal=True)

# Simulate 10 years of daily weather data
sim_times, sim_data = met_vg.simulate(T=10*365)

# Visualize results
met_vg.plot_meteogram_daily()
met_vg.plot_autocorr()
```

See the `scripts/` directory for more advanced examples.

## Running Tests

To run the test suite:

```bash
uv run pytest
```

Or install test dependencies and run:

```bash
uv sync --group test
uv run pytest
```

## Documentation

The documentation can be accessed online at
<https://vg-doc.readthedocs.io>.

<!-- The source package also ships with the sphinx-based documentation source -->
<!-- in the `doc` folder. Having [sphinx](sphinx.pocoo.org) installed, it can -->
<!-- be built by typing: -->

<!--     make html -->

<!-- inside the `doc` folder. -->

## Release notes

### 1.4

See [GitHub Releases](https://github.com/iskur/vg/releases) for detailed release notes.

### 1.3

- Prior to simulation, missing values can be infilled with the help of a VAR process.
- Option to phase-randomize VAR-residuals for better reproduction of low-frequency variability.
- Behind the scenes: VG is capable to be orchestrated by the soon-to-be released copula-based WeatherCop to produce multi-site data.
- Migration to modern build system with pyproject.toml
- Dependency management with uv

**Requirements:** Python ≥ 3.13

### 1.2

- Scenarios can be guided through changes in a variable that is not
  normally distributed.
- Disaggregation recreates seasonal changes in daily cycles.
- All scipy.stats.distributions can be used to fit variables.
- Bugfixes when disaggregating in the presence of nans.

### 1.1

This release makes VG more tolerant to \"dirty\" input data.

- Non-evenly spaced time series are allowed. WARNING: linear
  interpolation is used to regularize the data set.
- Gaps/NaNs are allowed. They are not filled in by linear interpolation,
  but actively ignored by the estimators.
- Disaggregation works on variables that have lower and/or upper bounds.

### 1.0

Initial release.

## Web sites

Code is hosted at: <https://github.com/iskur/vg/>

## License information

See the file \"LICENSE\" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
