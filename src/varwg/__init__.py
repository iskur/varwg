__all__ = [
    "meteo",
    "time_series_analysis",
    "helpers",
    "smoothing",
    "times",
    "ctimes",
    "ecdf",
    "sample_met",
]

from pathlib import Path
from dill import Pickler, Unpickler
import shelve
from numpy.random import default_rng
import matplotlib.pyplot as plt

from .core.core import VG, read_met
from .core import core, base, plotting
from . import times, ctimes

# Path to sample meteorological data file
sample_met = Path(__file__).parent / "sample.met"

shelve.Pickler = Pickler
shelve.Unpickler = Unpickler

rng = default_rng()


def reseed(seed):
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(seed).state


def set_conf(conf_obj):
    global conf
    conf = core.conf = base.conf = plotting.conf = conf_obj
