"""
Created on 26.02.2014

@author: dirk
"""

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    TestCase,
)
import varwg
from varwg.time_series_analysis import rain_stats


class Test(TestCase):
    def setUp(self):
        self.sequences = 10
        self.slen = 3
        self.n = self.sequences * self.slen
        rain_ = np.zeros(self.n).reshape(self.sequences, self.slen)
        rain_[::2] = 1
        self.rain = rain_.ravel()

    def tearDown(self):
        pass

    def test_trans_prob(self):
        pp = rain_stats.trans_prob(self.rain)
        p11 = p00 = (self.slen - 1.0) * self.sequences / (2 * self.n)
        p10 = self.sequences / (2.0 * self.n)
        p01 = (self.sequences / 2.0 - 1) / self.n
        pp_exp = np.array([[p00, p01], [p10, p11]])
        assert_almost_equal(pp_exp, pp)

    def test_spells(self):
        dry, wet = rain_stats.spell_lengths(self.rain)
        expected = np.full(int(self.sequences / 2), 3)
        assert_equal(dry, expected)
        assert_equal(wet, expected)

    def test_richardson_model(self):
        varwg.reseed(0)
        # trans_probs = np.array([[0.33020467, 0.16190849],
        #                         [0.16177813, 0.34597836]])
        # trans_probs = np.array([[0.330, 0.162],
        #                         [0.162, 0.346]])
        trans_probs = np.array([[0.613, 0.145], [0.145, 0.096]])
        occurrences = rain_stats.richardson_model_occ(1e6, trans_probs)
        emp_trans_probs = rain_stats.trans_prob(occurrences)
        assert_almost_equal(trans_probs, emp_trans_probs, 3)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    # run_module_suite()
    pass
