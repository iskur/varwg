import numpy as np
from numpy.testing import assert_almost_equal, TestCase
from vg import smoothing


class Test(TestCase):

    def setUp(self):
        self.data = np.array(10 * [-1.0] + 10 * [0] + 10 * [1])
        self.window_len = 5

    def test_max(self):
        max_test = smoothing.max(self.data, self.window_len)
        max_exp = np.array(
            (10 - self.window_len) * [-1.0]
            + 10 * [0]
            + (10 + self.window_len) * [1]
        )
        assert_almost_equal(max_test, max_exp)

    def test_max_nofuture(self):
        max_test = smoothing.max(self.data, self.window_len, no_future=True)
        assert_almost_equal(max_test, self.data)


if __name__ == "__main__":
    pass
    # run_module_suite()
