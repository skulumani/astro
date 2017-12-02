from astro import tle, satellite
import numpy as np
import os

cwd = os.path.realpath(os.path.dirname(__file__))

class TestISSUSAFA():
    l0 = "0 ISS (ZARYA)             "
    l1 = "1 25544U 98067A   06164.43693594  .00014277  00000-0  10780-3 0  6795"
    l2 = "2 25544  51.6455 280.1294 0004346 245.9311 226.9658 15.72751720375095"
    ifile = 'Predict.dat'
    sat = tle.get_tle(os.path.join(cwd, ifile))

    def test_validtle(self):
        np.testing.assert_equal(tle.validtle(self.l0, self.l1, self.l2), True)

    def test_line1_checksum(self):
        np.testing.assert_allclose(tle.checksum(self.l1), 5)

    def test_line2_checksum(self):
        np.testing.assert_allclose(tle.checksum(self.l2), 5)

    def test_epoch_year(self):
        np.testing.assert_allclose(self.sat[0].epoch_year, 2006)

    def test_epoch_day(self):
        np.testing.assert_allclose(self.sat[0].epoch_day, 164.43693594)

    def test_ndot2_revperdaysquared(self):
        np.testing.assert_allclose(self.sat[0].tle.ndot_over_2, 0.00014277)

    def test_inclination_deg(self):
        np.testing.assert_allclose(self.sat[0].tle.inc, 51.64550000)

    def test_raan_deg(self):
        np.testing.assert_allclose(self.sat[0].tle.raan, 280.12940000)

    def test_ecc_deg(self):
        np.testing.assert_allclose(self.sat[0].tle.ecc, 0.00043460)

    def test_argp_deg(self):
        np.testing.assert_allclose(self.sat[0].tle.argp, 245.93110000)

    def test_mean_anomaly_deg(self):
        np.testing.assert_allclose(self.sat[0].tle.ma, 226.9658000)

    def test_mean_motion_deg(self):
        np.testing.assert_allclose(self.sat[0].tle.mean_motion, 15.72751720)

    def test_mean_motion_rad(self):
        np.testing.assert_allclose(self.sat[0].n0, 0.00114373733)

    def test_ecc(self):
        np.testing.assert_allclose(self.sat[0].ecc0, 0.00043460000)

    def test_inc_rad(self):
        np.testing.assert_allclose(self.sat[0].inc0, 0.90138401884)

    def test_raan_rad(self):
        np.testing.assert_allclose(self.sat[0].raan0, 4.88918036164)

    def test_argp_rad(self):
        np.testing.assert_allclose(self.sat[0].argp0, 4.29230742805)

    def test_ndot2_radpersecsquared(self):
        np.testing.assert_allclose(self.sat[0].ndot2, 1.20168141063e-013)

    def test_eccdot_persecond(self):
        np.testing.assert_allclose(self.sat[0].eccdot, -1.40011545218e-10)

    def test_mean_anomaly_rad(self):
        np.testing.assert_allclose(self.sat[0].mean0, 3.96130049942e0)

    def test_raandot_radpersecond(self):
        np.testing.assert_allclose(self.sat[0].raandot, -1.03554877709e-6)

    def test_argpdot_radpersecond(self):
        np.testing.assert_allclose(self.sat[0].argpdot, 7.72047261206e-7)
