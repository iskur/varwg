import os
import sys
import shutil
from pathlib import Path
import importlib.util

import varwg

filepath = Path(varwg.__file__).parent / "core/tests/test_varwg.py"
# filepath = Path("../varwg/core/tests/test_varwg.py").absolute()
module_name = "test_varwg"
spec = importlib.util.spec_from_file_location(module_name, filepath)
test_varwg = importlib.util.module_from_spec(spec)
sys.modules[module_name] = test_varwg
spec.loader.exec_module(test_varwg)

# config_template = varwg.config_template
# varwg.set_conf(config_template)

import config_konstanz as conf

varwg.set_conf(conf)

# py_version = sys.version_info.major
# sample_met_filepath = ("/home/dirk/workspace/python/varwg/varwg/sample_py%d.met" %
#                        py_version)
# sample_met_filepath = "/home/dirk/workspace/python/varwg/varwg/sample.met"
sample_met_filepath = test_varwg.met_file


def main(T=None):
    met_varwg: varwg.VG = varwg.VG(
        test_varwg.var_names,
        # refit="R",
        refit=True,
        verbose=True,
        infill=True,
        fit_kwds=test_varwg.fit_kwds,
    )
    # note: this has not to be the same p as in the tests
    met_varwg.fit(**test_varwg.fit_kwds)
    varwg.reseed(test_varwg.seed)
    met_varwg.simulate(T=test_varwg.T)  # , theta_incr=-2)
    met_varwg.disaggregate(test_varwg.disagg_varnames)

    met_varwg.plot_meteogram_daily()
    varwg.plt.show()

    met_varwg.to_df("hourly output", with_conversions=False).to_csv(
        met_varwg.outfilepath, sep="\t"
    )
    # shutil.copy(met_varwg.outfilepath, sample_met_filepath)

    return met_varwg.outfilepath, met_varwg


# relative humidity exceeds its bounds during disaggregation
# rh_ii = var_names.index("rh")
# sim_dis[rh_ii] = sim[rh_ii, :-1].repeat(24)
#
# print "Writing the sample.met file"
# with open(sample_met_filepath, "w") as met_file:
#     met_file.write("time\t%s\n" % "\t".join(met_varwg.var_names))
#     for dtime, arr in zip(dtimes, sim_dis.T):
#         met_file.write("%s\t" % dtime.isoformat())
#         met_file.write("\t".join(["%.2f" % val for val in arr]) + "\n")

# print "Testing whether we can use it as VG input."
# varwg.conf.seasonal_cache_file = "/tmp/seasonal_fit.she"
# test_varwg = varwg.VG(var_names, met_file=sample_met_filepath)
# test_varwg.fit(3)
# dtimes, sim = test_varwg.simulate()

if __name__ == "__main__":
    import warnings

    warnings.simplefilter("error", RuntimeWarning)

    outfilepath, met_varwg = main(3 * 365)

    shutil.copy(outfilepath, sample_met_filepath)
