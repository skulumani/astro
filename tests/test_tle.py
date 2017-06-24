"""Test the TLE parsing and transformations
"""
import numpy as np
from .. import tle
import pdb
import os

cwd = os.path.realpath(os.path.dirname(__file__))

class TestTLEISS():
    l0 = "0 ISS (ZARYA)"
    l1 = "1 25544U 98067A   17168.12419852 +.00002055 +00000-0 +38402-4 0  9998"
    l2 = "2 25544 051.6427 038.6434 0004440 287.0211 207.7840 15.54043832061749"
    
    elements = tle.parsetle(l0, l1, l2)

    def test_validtle(self):
        np.testing.assert_equal(tle.validtle(self.l0, self.l1, self.l2), True)

    def test_line1_checksum(self):
        np.testing.assert_equal(tle.checksum(self.l1), 8)

    def test_line2_checksum(self):
        np.testing.assert_equal(tle.checksum(self.l2), 9)

    def test_satname(self):
        np.testing.assert_string_equal(self.elements.satname, "0 ISS (ZARYA)")

    def test_satnum(self):
        np.testing.assert_equal(self.elements.satnum, 25544)

    def test_classification(self):
        np.testing.assert_string_equal(self.elements.classification, 'U')

    def test_id_year(self):
        np.testing.assert_allclose(self.elements.id_year, 98)

    def test_id_launch(self):
        np.testing.assert_allclose(self.elements.id_launch, 67)

    def test_id_piece(self):
        np.testing.assert_string_equal(self.elements.id_piece, 'A  ')

    def test_epoch_year(self):
        np.testing.assert_allclose(self.elements.epoch_year, 17)

    def test_epoch_day(self):
        np.testing.assert_allclose(self.elements.epoch_day, 168.12419852)

    def test_ndot_over_2(self):
        np.testing.assert_allclose(self.elements.ndot_over_2, 0.00002055)
        
    def test_nddot_over_6(self):
        np.testing.assert_allclose(self.elements.nddot_over_6, 0)

    def test_bstar(self):
        np.testing.assert_allclose(self.elements.bstar, 38402e-4/1e5)

    def test_ephtype(self):
        np.testing.assert_allclose(self.elements.ephtype, 0)

    def test_elnum(self):
        np.testing.assert_allclose(self.elements.elnum, 999)

    def test_checksum1(self):
        np.testing.assert_allclose(self.elements.checksum1, 8)

    def test_inc(self):
        np.testing.assert_allclose(self.elements.inc, 51.6427)

    def test_raan(self):
        np.testing.assert_allclose(self.elements.raan, 38.6434)

    def test_ecc(self):
        np.testing.assert_allclose(self.elements.ecc, 0.0004440)

    def test_argp(self):
        np.testing.assert_allclose(self.elements.argp, 287.0211)

    def test_ma(self):
        np.testing.assert_allclose(self.elements.ma, 207.7840)

    def test_mean_motion(self):
        np.testing.assert_allclose(self.elements.mean_motion, 15.54043832)

    def test_epoch_rev(self):
        np.testing.assert_allclose(self.elements.epoch_rev, 6174)

    def test_checksum2(self):
        np.testing.assert_allclose(self.elements.checksum2, 9)


class TestTLESL3():
    l0 = "0 SL-3 R/B"
    l1 = "1 00056U 60011  B 60267.06075999  .41473555 +00000-0 +00000-0 0  9999"
    l2 = "2 00056 064.9400 008.3000 0007000 015.2810 344.7297 16.37352814005552"
    
    elements = tle.parsetle(l0, l1, l2)

    def test_validtle(self):
        np.testing.assert_equal(tle.validtle(self.l0, self.l1, self.l2), True)

    def test_line1_checksum(self):
        np.testing.assert_equal(tle.checksum(self.l1), 9)

    def test_line2_checksum(self):
        np.testing.assert_equal(tle.checksum(self.l2), 2)

    def test_satname(self):
        np.testing.assert_string_equal(self.elements.satname, "0 SL-3 R/B")

    def test_satnum(self):
        np.testing.assert_equal(self.elements.satnum, 56)

    def test_classification(self):
        np.testing.assert_string_equal(self.elements.classification, 'U')

    def test_id_year(self):
        np.testing.assert_allclose(self.elements.id_year, 60)

    def test_id_launch(self):
        np.testing.assert_allclose(self.elements.id_launch, 11 )

    def test_id_piece(self):
        np.testing.assert_string_equal(self.elements.id_piece, '  B')

    def test_epoch_year(self):
        np.testing.assert_allclose(self.elements.epoch_year, 60)

    def test_epoch_day(self):
        np.testing.assert_allclose(self.elements.epoch_day, 267.06075999)

    def test_ndot_over_2(self):
        np.testing.assert_allclose(self.elements.ndot_over_2, 0.41473555)
        
    def test_nddot_over_6(self):
        np.testing.assert_allclose(self.elements.nddot_over_6, 0)

    def test_bstar(self):
        np.testing.assert_allclose(self.elements.bstar, 0)

    def test_ephtype(self):
        np.testing.assert_allclose(self.elements.ephtype, 0)

    def test_elnum(self):
        np.testing.assert_allclose(self.elements.elnum, 999)

    def test_checksum1(self):
        np.testing.assert_allclose(self.elements.checksum1, 9)

    def test_inc(self):
        np.testing.assert_allclose(self.elements.inc, 64.9400)

    def test_raan(self):
        np.testing.assert_allclose(self.elements.raan, 8.300)

    def test_ecc(self):
        np.testing.assert_allclose(self.elements.ecc, 0.0007000)

    def test_argp(self):
        np.testing.assert_allclose(self.elements.argp, 15.2810)

    def test_ma(self):
        np.testing.assert_allclose(self.elements.ma, 344.7297)

    def test_mean_motion(self):
        np.testing.assert_allclose(self.elements.mean_motion, 16.373528140)

    def test_epoch_rev(self):
        np.testing.assert_allclose(self.elements.epoch_rev, 555)

    def test_checksum2(self):
        np.testing.assert_allclose(self.elements.checksum2, 2)


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


# class TestTLEVallado():
#     l0 = "Vallado Pg 113"
#     l1 = "1 16609U 86017A   93352.53502934  .00007889  00000-0  10529-3 0   342"
#     l2 = "2 16609  51.6190  13.3340 0005770 102.5680 257.5950 15.59114070447869"
# 
#     def test_line1_checksum(self):
#         np.testing.assert_allclose(tle.checksum(self.l1), 2)
# 
#     def test_line_checksum(self):
#         np.testing.assert_allclose(tle.checksum(self.l2), 9)
# 
#     def test_first_example_tle_line1(self):
#         l1 = "1 16609U 86017A   92048.88339427  .00000000      0 0  51786-4 0  7632"
#         np.testing.assert_allclose(tle.checksum(l1), 2)
# 
#     def test_first_example_tle_line2(self):
#         l2 = "2 16609  51.6016 131.8518 0014007 157.0182 203.1411 15.62321407     9"
#         np.testing.assert_allclose(tle.checksum(l2), 9)
# 
#     def test_second_example_tle_line1(self):
#         l1 = "1    50U 60009B   17095.28396119 -.00000095 +00000-0 -81132-4 0     6"
#         np.testing.assert_allclose(tle.checksum(l1), 6)
# 
#     def test_second_example_tle_line2(self):
#         l2 = "2    50  47.2298 149.9498 0112984 081.3681 279.9890 12.20099583     3"
#         np.testing.assert_allclose(tle.checksum(l2), 3)





