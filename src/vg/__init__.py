__all__ = [
    "meteo",
    "time_series_analysis",
    "helpers",
    "smoothing",
    "times",
    "ctimes",
]

from dill import Pickler, Unpickler
import shelve
from numpy.random import default_rng
import matplotlib.pyplot as plt

from .core.vg_core import VG, read_met
from .core import vg_core, vg_base, vg_plotting

shelve.Pickler = Pickler
shelve.Unpickler = Unpickler

rng = default_rng()


def reseed(seed):
    BitGen = type(rng.bit_generator)
    rng.bit_generator.state = BitGen(seed).state


def set_conf(conf_obj):
    global conf
    conf = vg_core.conf = vg_base.conf = vg_plotting.conf = conf_obj
