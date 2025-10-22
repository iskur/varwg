# VARWG: Vector Autoregressive Weather Generator

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/varwg/badge/?version=latest)](https://vg-doc.readthedocs.io)
[![Managed with uv](https://img.shields.io/badge/managed_with-uv-blue)](https://github.com/astral-sh/uv)
![Last Commit](https://img.shields.io/github/last-commit/iskur/varwg)

## What is VARWG?

VARWG is a single-site Vector-Autoregressive weather generator that was developed for hydrodynamic and ecologic modelling of lakes. It includes a number of possibilities to define climate scenarios. For example, changes in mean or in the variability of air temperature can be set. Correlations during simulations are preserved, so that these changes propagate from the air temperature to the other simulated variables.

## About the Name Change

The project was renamed from **VG** to **VARWG** because there is already a different package named `vg` on PyPI. To avoid conflicts and ensure the package is properly discoverable on PyPI, we adopted the more descriptive name **VARWG** (Vector-Autoregressive Weather Generator). For backward compatibility, the old `VG` class name is still available as an alias, so existing code will continue to work without modifications.

## Installation

### From PyPI (Recommended)

```bash
pip install varwg
```

Pre-built wheels are available for:
- **Linux**: x86_64
- **Windows**: AMD64
- **macOS**: x86_64 (Intel), arm64 (Apple Silicon)

No compiler needed! If a wheel isn't available for your platform, pip will automatically build from source (requires C compiler and Cython).

### From Source

```bash
git clone https://github.com/iskur/varwg.git
cd varwg
pip install -e .
```

Building from source requires:
- C compiler (gcc/clang/MSVC)
- Cython >= 3.1.1
- NumPy >= 1.26.0

## Quick Start

After installation, you can use VARWG to generate synthetic weather data:

```python
import varwg as vg

# Configure VARWG with default settings
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

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes, or view [releases on GitHub](https://github.com/iskur/varwg/releases).

**Current version: 1.4.0** - Python ≥ 3.13 required

## Web sites

Code is hosted at: <https://github.com/iskur/varwg/>

## Citation

If you use VARWG in your research, please cite:

**Schlabing, D., Frassl, M. A., Eder, M., Rinke, K., & Bárdossy, A. (2014).** Use of a weather generator for simulating climate change effects on ecosystems: A case study on Lake Constance. *Environmental Modelling & Software*, 61, 326-338. https://doi.org/10.1016/j.envsoft.2014.06.028

### BibTeX

```bibtex
@article{schlabing2014vg,
  author = {Schlabing, Dirk and Frassl, Marieke A. and Eder, Magdalena and Rinke, Karsten and B{\'a}rdossy, Andr{\'a}s},
  title = {Use of a weather generator for simulating climate change effects on ecosystems: A case study on {Lake Constance}},
  journal = {Environmental Modelling \& Software},
  volume = {61},
  pages = {326--338},
  year = {2014},
  doi = {10.1016/j.envsoft.2014.02.028},
  url = {https://doi.org/10.1016/j.envsoft.2014.06.028}
}
```

## Publications Using VARWG

The following publications have used VARWG for weather generation:

- **Kobler, U. G., Wüest, A., & Schmid, M. (2018).** Effects of Lake–Reservoir Pumped-Storage Operations on Temperature and Water Quality. *Sustainability*, 10(6), 1968. https://doi.org/10.3390/su10061968

- **Fenocchi, A., Petaccia, G., Sibilla, S., & Dresti, C. (2018).** Forecasting the evolution in the mixing regime of a deep subalpine lake under climate change scenarios through numerical modelling (Lake Maggiore, Northern Italy/Southern Switzerland). *Climate Dynamics*, 51, 3521-3536. https://doi.org/10.1007/s00382-018-4094-6

- **Gal, G., Gilboa, Y., Schachar, N., Estroti, M., & Schlabing, D. (2020).** Ensemble Modeling of the Impact of Climate Warming and Increased Frequency of Extreme Climatic Events on the Thermal Characteristics of a Sub-Tropical Lake. *Water*, 12(7), 1982. https://doi.org/10.3390/w12071982

- **Eder, M. (2013).** *Climate sensitivity of a large lake*. PhD Thesis, University of Stuttgart, http://dx.doi.org/10.18419/opus-509.

*If you've published work using VARWG, please let us know by [opening an issue](https://github.com/iskur/varwg/issues) so we can add it to this list!*

## License information

See the file \"LICENSE\" for information on the history of this software, terms & conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
