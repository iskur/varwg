"""Generates data/test_out_sample.met using ../sample.met as input."""

import os
import shutil
import tempfile
import vg
from vg.core.tests import test_vg

config_template = vg.config_template
vg.set_conf(config_template)

# import config_konstanz as conf
# vg.set_conf(conf)


p, T = test_vg.p, test_vg.T


def main():
    data_filepath = os.path.join(
        os.path.dirname(vg.core.__file__), "tests", "data"
    )
    test_in_data_filepath = os.path.join(data_filepath, "sample.met")
    if not os.path.exists(test_in_data_filepath):
        # try to look where the stand-alone vg has its sample data
        test_in_data_filepath = os.path.join(
            "..", os.path.dirname(vg.__file__), "sample.met"
        )
    test_out_data_filepath = os.path.join(data_filepath, "test_out_sample.met")
    # in order not to refit the present fit...
    cache_dir = tempfile.mkdtemp("vg_test_data_gen")

    vg.reseed(test_vg.seed)
    met_vg = vg.VG(
        test_vg.var_names,
        met_file=test_in_data_filepath,
        cache_dir=cache_dir,
        data_dir=cache_dir,
        refit=True,
        verbose=True,
        infill=True,
        rain_method="regression",
    )
    met_vg.fit(p)
    vg.reseed(test_vg.seed)
    met_vg.simulate(T=T)
    met_vg.disaggregate(test_vg.disagg_varnames)
    # shutil.copy(met_vg.outfilepath, test_out_data_filepath)
    # met_vg.to_df("hourly output", with_conversions=False).to_csv(
    #     test_out_data_filepath, sep="\t"
    # )
    met_vg.to_df("hourly output", with_conversions=True).to_csv(
        test_out_data_filepath, sep="\t"
    )
    shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main()
