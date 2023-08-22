import os
import shutil
import tempfile
import warnings

import pandas as pd
import numpy as np
from builtins import range
from numpy.testing import TestCase, assert_almost_equal, dec, run_module_suite

import vg
from vg.core import monty


vg.conf = vg.config_template


class Test(TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp("vg_temporary_test")

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    @dec.slow
    def test_Monty(self):
        """Do we get the same results sequentially and in parallel?"""
        import sys

        if sys.version_info.major == 2:
            warnings.warn("Not supported in Python 2")
            return
        n_realizations = 2
        vg_kwds = dict(var_names=("theta", "ILWR", "Qsw", "rh", "u", "v"))
        fit_kwds = dict(p=3)
        sim_kwds = dict(T=100)
        dis_kwds = {}

        outfilepath_par = os.path.join(self.data_dir, "parallel_%d.csv")
        with monty.Monty(2, conf=vg.config_template) as mon:
            mon.init(vg_kwds=vg_kwds, fit_kwds=fit_kwds)
            mon.run(
                n_realizations,
                use_seed=True,
                sim_kwds=sim_kwds,
                dis_kwds=dis_kwds,
                glm_kwds=dict(outfilepath=outfilepath_par),
            )

        outfilepath_seq = os.path.join(self.data_dir, "sequential_%d.csv")
        met_vg = vg.VG(**vg_kwds)
        met_vg.fit(**fit_kwds)
        for ri in range(n_realizations):
            vg.reseed(ri)
            met_vg.simulate(**sim_kwds)
            met_vg.disaggregate(**dis_kwds)
            met_vg.to_glm(outfilepath_seq % ri)

        for ri in range(n_realizations):
            seq = pd.read_csv(
                outfilepath_seq % ri, parse_dates=True, index_col=0
            )
            par = pd.read_csv(
                outfilepath_par % ri, parse_dates=True, index_col=0
            )
            assert_almost_equal(par.as_matrix(), seq.as_matrix())


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    run_module_suite()
