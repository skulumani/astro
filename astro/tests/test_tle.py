"""Test the TLE parsing and transformations
"""
import numpy as np
from astro import tle
import pdb


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

class TestTLEKOREASAT():
    l0 = "0 KOREASAT 5A"
    l1 = "1 42984U 17066A   17327.72549185 -.00000354  00000-0  00000+0 0  9990"
    l2 = "2 42984   0.0495 103.7501 0000074 356.8166 338.0747  1.00270863   278"
    
    elements = tle.parsetle(l0, l1, l2)

    def test_validtle(self):
        np.testing.assert_equal(tle.validtle(self.l0, self.l1, self.l2), True)

    def test_line1_checksum(self):
        np.testing.assert_equal(tle.checksum(self.l1), 0)

    def test_line2_checksum(self):
        np.testing.assert_equal(tle.checksum(self.l2), 8)

    def test_satname(self):
        np.testing.assert_string_equal(self.elements.satname, "0 KOREASAT 5A")

    def test_satnum(self):
        np.testing.assert_equal(self.elements.satnum, 42984)

    def test_classification(self):
        np.testing.assert_string_equal(self.elements.classification, 'U')

    def test_id_year(self):
        np.testing.assert_allclose(self.elements.id_year, 17)

    def test_id_launch(self):
        np.testing.assert_allclose(self.elements.id_launch, 66)

    def test_id_piece(self):
        np.testing.assert_string_equal(self.elements.id_piece, 'A  ')

    def test_epoch_year(self):
        np.testing.assert_allclose(self.elements.epoch_year, 17)

    def test_epoch_day(self):
        np.testing.assert_allclose(self.elements.epoch_day, 327.72549185)

    def test_ndot_over_2(self):
        np.testing.assert_allclose(self.elements.ndot_over_2, -0.00000354)
        
    def test_nddot_over_6(self):
        np.testing.assert_allclose(self.elements.nddot_over_6, 0)

    def test_bstar(self):
        np.testing.assert_allclose(self.elements.bstar, 0)

    def test_ephtype(self):
        np.testing.assert_allclose(self.elements.ephtype, 0)

    def test_elnum(self):
        np.testing.assert_allclose(self.elements.elnum, 999)

    def test_checksum1(self):
        np.testing.assert_allclose(self.elements.checksum1, 0)

    def test_inc(self):
        np.testing.assert_allclose(self.elements.inc, 0.0495)

    def test_raan(self):
        np.testing.assert_allclose(self.elements.raan, 103.7501)

    def test_ecc(self):
        np.testing.assert_allclose(self.elements.ecc, 0.0000074)

    def test_argp(self):
        np.testing.assert_allclose(self.elements.argp, 356.8166)

    def test_ma(self):
        np.testing.assert_allclose(self.elements.ma, 338.0747)

    def test_mean_motion(self):
        np.testing.assert_allclose(self.elements.mean_motion, 1.00270863)

    def test_epoch_rev(self):
        np.testing.assert_allclose(self.elements.epoch_rev, 27)

    def test_checksum2(self):
        np.testing.assert_allclose(self.elements.checksum2, 8)


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

class TestParser():
    ofile_actual, sl_actual = tle.parse_args(['stations', 'tle.txt'])
    
    def test_sat_list(self):
        np.testing.assert_string_equal(self.sl_actual, 'stations')

    def test_ofile(self):
        np.testing.assert_string_equal(self.ofile_actual, 'tle.txt')

