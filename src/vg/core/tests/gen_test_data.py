"""Generates data/test_out_sample.met using ../sample.met as input."""

import shutil
import tempfile

import vg
from pathlib import Path
from vg.core.tests import test_vg

config_template = vg.config_template
vg.set_conf(config_template)

T = test_vg.T
fit_kwds = test_vg.fit_kwds


def main():
    data_filepath = Path(vg.core.__file__).parent / "tests" / "data"
    test_in_data_filepath = data_filepath / "sample.met"
    if not test_in_data_filepath.exists():
        # try to look where the stand-alone vg has its sample data
        test_in_data_filepath = (
            Path("..") / Path(vg.__file__).parent / "sample.met"
        )
    test_out_data_filepath = data_filepath / "test_out_sample.met"
    # in order not to refit the present fit...
    cache_dir = tempfile.mkdtemp("vg_test_data_gen")

    vg.reseed(test_vg.seed)
    met_vg: vg.VG = vg.VG(
        test_vg.var_names,
        met_file=test_in_data_filepath,
        cache_dir=cache_dir,
        data_dir=cache_dir,
        refit=True,
        verbose=True,
        infill=True,
        rain_method="regression",
    )
    met_vg.fit(**fit_kwds)
    vg.reseed(test_vg.seed)
    met_vg.simulate(T=T)
    sim = met_vg.disaggregate(test_vg.disagg_varnames)[1]
    met_vg.to_df("hourly output", with_conversions=True).to_csv(
        test_out_data_filepath, sep="\t"
    )

    # do a roundtrip test
    met = vg.read_met(
        test_out_data_filepath, verbose=True, with_conversions=True
    )[1]
    import numpy as np
    import numpy.testing as npt

    sample_sim = np.array([met[var_name] for var_name in test_vg.var_names])
    npt.assert_almost_equal(sim, sample_sim)

    shutil.rmtree(cache_dir)

    return met_vg


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    met_vg = main()
    met_vg.plot_meteogram_daily()
    plt.show()
