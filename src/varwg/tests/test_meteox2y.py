import numpy.testing as npt
from varwg.meteo import meteox2y


class Test(npt.TestCase):

    def test_humidity_roundtrip(self):
        rh = [0.9, 0.8, 0.7, 0.6, 0.5]
        at = [0.0, 2.5, 5.0, 7.5, 10.0]
        ah = meteox2y.rel2abs_hum(rh, at)
        rh_back = meteox2y.abs_hum2rel(ah, at)
        npt.assert_almost_equal(rh_back, rh)
