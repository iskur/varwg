import os
import sys
import shutil
from pathlib import Path
import importlib.util

import vg

filepath = Path(vg.__file__).parent / "core/tests/test_vg.py"
# filepath = Path("../vg/core/tests/test_vg.py").absolute()
module_name = "test_vg"
spec = importlib.util.spec_from_file_location(module_name, filepath)
test_vg = importlib.util.module_from_spec(spec)
sys.modules[module_name] = test_vg
spec.loader.exec_module(test_vg)

# config_template = vg.config_template
# vg.set_conf(config_template)

import config_konstanz as conf

vg.set_conf(conf)

# py_version = sys.version_info.major
# sample_met_filepath = ("/home/dirk/workspace/python/vg/vg/sample_py%d.met" %
#                        py_version)
# sample_met_filepath = "/home/dirk/workspace/python/vg/vg/sample.met"
sample_met_filepath = test_vg.met_file


def main(T=None):
    met_vg: vg.VG = vg.VG(
        test_vg.var_names,
        # refit="R",
        refit=True,
        verbose=True,
        infill=True,
        fit_kwds=test_vg.fit_kwds,
    )
    # note: this has not to be the same p as in the tests
    met_vg.fit(**test_vg.fit_kwds)
    vg.reseed(test_vg.seed)
    met_vg.simulate(T=test_vg.T)  # , theta_incr=-2)
    met_vg.disaggregate(test_vg.disagg_varnames)

    met_vg.plot_meteogram_daily()
    vg.plt.show()

    met_vg.to_df("hourly output", with_conversions=False).to_csv(
        met_vg.outfilepath, sep="\t"
    )
    # shutil.copy(met_vg.outfilepath, sample_met_filepath)

    return met_vg.outfilepath, met_vg


# relative humidity exceeds its bounds during disaggregation
# rh_ii = var_names.index("rh")
# sim_dis[rh_ii] = sim[rh_ii, :-1].repeat(24)
#
# print "Writing the sample.met file"
# with open(sample_met_filepath, "w") as met_file:
#     met_file.write("time\t%s\n" % "\t".join(met_vg.var_names))
#     for dtime, arr in zip(dtimes, sim_dis.T):
#         met_file.write("%s\t" % dtime.isoformat())
#         met_file.write("\t".join(["%.2f" % val for val in arr]) + "\n")

# print "Testing whether we can use it as VG input."
# vg.conf.seasonal_cache_file = "/tmp/seasonal_fit.she"
# test_vg = vg.VG(var_names, met_file=sample_met_filepath)
# test_vg.fit(3)
# dtimes, sim = test_vg.simulate()

if __name__ == "__main__":
    import warnings

    warnings.simplefilter("error", RuntimeWarning)

    outfilepath, met_vg = main(3 * 365)

    shutil.copy(outfilepath, sample_met_filepath)
