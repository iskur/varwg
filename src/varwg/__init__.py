__all__ = [
    "meteo",
    "time_series_analysis",
    "helpers",
    "smoothing",
    "times",
    "ctimes",
    "ecdf",
    "sample_met",
    "get_rng",
    "reseed",
    "rng",
]

from pathlib import Path
from dill import Pickler, Unpickler
import shelve
from numpy.random import default_rng
import matplotlib.pyplot as plt
import threading

from .core.core import VarWG, read_met
from .core import core, base, plotting
from . import times, ctimes

# Backward compatibility aliases
VG = VarWG
VGBase = base.VGBase
VGPlotting = plotting.VGPlotting

# Path to sample meteorological data file
sample_met = Path(__file__).parent / "sample.met"

shelve.Pickler = Pickler
shelve.Unpickler = Unpickler

# Thread-local storage for RNG
_thread_rng = threading.local()

# Keep old rng for backwards compatibility (deprecated)
# WARNING: Direct use of varwg.rng is deprecated and not thread-safe.
# Use varwg.get_rng() instead for thread-safe random number generation.
rng = default_rng()


def get_rng():
    """Get thread-local RNG generator.

    Each thread gets its own independent Generator instance.
    First call in a thread creates a new Generator; subsequent
    calls return the same one.

    Returns
    -------
    numpy.random.Generator
        Thread-local random number generator
    """
    if not hasattr(_thread_rng, 'rng'):
        _thread_rng.rng = default_rng()
    return _thread_rng.rng


def reseed(seed):
    """Seed the thread-local RNG.

    Only affects the RNG for the current thread.

    Parameters
    ----------
    seed : int
        Seed value for the thread's RNG
    """
    rng_instance = get_rng()
    BitGen = type(rng_instance.bit_generator)
    rng_instance.bit_generator.state = BitGen(seed).state


def set_conf(conf_obj):
    """Set the global configuration object for all VarWG modules.

    Parameters
    ----------
    conf_obj : module
        A configuration module with attributes like 'dists', 'latitude', etc.

    Raises
    ------
    ValueError
        If conf_obj is missing critical attributes required by VarWG.
    """
    # Validate that conf_obj has required attributes
    required_attrs = ['dists', 'dists_kwds', 'latitude', 'longitude']
    missing = [attr for attr in required_attrs if not hasattr(conf_obj, attr)]
    if missing:
        raise ValueError(
            f"Configuration object missing required attributes: {missing}. "
            f"Provided: {type(conf_obj).__name__} from {getattr(conf_obj, '__file__', 'unknown')}"
        )

    global conf
    conf = core.conf = base.conf = plotting.conf = conf_obj
