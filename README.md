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

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes, or view [releases on GitHub](https://github.com/iskur/vg/releases).

**Current version: 1.4.0** - Python ≥ 3.13 required

## Web sites

Code is hosted at: <https://github.com/iskur/vg/>

## License information

See the file \"LICENSE\" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.
