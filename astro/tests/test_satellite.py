from astro import tle, satellite, time, predict
import numpy as np
import os
import pytest
import tempfile
import filecmp

cwd = os.path.realpath(os.path.dirname(__file__))

class TestISSUSAFA():
    

    l0 = "0 ISS (ZARYA)             "
    l1 = "1 25544U 98067A   06164.43693594  .00014277  00000-0  10780-3 0  6795"
    l2 = "2 25544  51.6455 280.1294 0004346 245.9311 226.9658 15.72751720375095"
    ifile = 'Predict.dat'

    # define site location

    lat = 39.006
    lon = -104.883
    alt = 2.184
    
    site_location = (lat, lon, alt)
    start_date = (2006, 6, 19, 0, 0, 0)
    end_date = (2006, 6, 29, 0, 0, 0)
    

    jd_start, _ = time.date2jd(start_date[0], start_date[1], start_date[2],
                            start_date[3], start_date[4], start_date[5])
    jd_end, _ = time.date2jd(end_date[0], end_date[1], end_date[2],
                            end_date[3], end_date[4], end_date[5])

    jd_step = 2 / (24 * 60)
    jd_span = np.arange(jd_start, jd_end, jd_step)

    site = predict.build_site(jd_span, np.deg2rad(site_location[0]),
                            np.deg2rad(site_location[1]),
                            site_location[2])
    
    sats = tle.get_tle(os.path.join(cwd, ifile))
    sat = sats[0]

    sat.tle_update(jd_span)
    sat.visible(site)
    
    r_eci = satellite.tle_update(sat, jd_span)
    all_passes = satellite.visible(sat.r_eci, site, jd_span)
    all_passes_parallel = satellite.parallel_predict(sat, jd_span, site)

    # output using the two functions
    class_output_file = os.path.join(tempfile.gettempdir(), 'output_class.txt')
    fcn_output_file = os.path.join(tempfile.gettempdir(), 'output_fcn.txt')
    
    # remove if they exist
    if os.path.isfile(class_output_file):
        os.remove(class_output_file)
    if os.path.isfile(fcn_output_file):
        os.remove(fcn_output_file)

    satellite.output(sat, all_passes, fcn_output_file)

    sat.output(class_output_file)
    

    def test_validtle(self):
        np.testing.assert_equal(tle.validtle(self.l0, self.l1, self.l2), True)

    def test_line1_checksum(self):
        np.testing.assert_allclose(tle.checksum(self.l1), 5)

    def test_line2_checksum(self):
        np.testing.assert_allclose(tle.checksum(self.l2), 5)

    def test_epoch_year(self):
        np.testing.assert_allclose(self.sat.epoch_year, 2006)

    def test_epoch_day(self):
        np.testing.assert_allclose(self.sat.epoch_day, 164.43693594)

    def test_ndot2_revperdaysquared(self):
        np.testing.assert_allclose(self.sat.tle.ndot_over_2, 0.00014277)

    def test_inclination_deg(self):
        np.testing.assert_allclose(self.sat.tle.inc, 51.64550000)

    def test_raan_deg(self):
        np.testing.assert_allclose(self.sat.tle.raan, 280.12940000)

    def test_ecc_deg(self):
        np.testing.assert_allclose(self.sat.tle.ecc, 0.00043460)

    def test_argp_deg(self):
        np.testing.assert_allclose(self.sat.tle.argp, 245.93110000)

    def test_mean_anomaly_deg(self):
        np.testing.assert_allclose(self.sat.tle.ma, 226.9658000)

    def test_mean_motion_deg(self):
        np.testing.assert_allclose(self.sat.tle.mean_motion, 15.72751720)

    def test_mean_motion_rad(self):
        np.testing.assert_allclose(self.sat.n0, 0.00114373733)

    def test_ecc(self):
        np.testing.assert_allclose(self.sat.ecc0, 0.00043460000)

    def test_inc_rad(self):
        np.testing.assert_allclose(self.sat.inc0, 0.90138401884)

    def test_raan_rad(self):
        np.testing.assert_allclose(self.sat.raan0, 4.88918036164)

    def test_argp_rad(self):
        np.testing.assert_allclose(self.sat.argp0, 4.29230742805)

    def test_ndot2_radpersecsquared(self):
        np.testing.assert_allclose(self.sat.ndot2, 1.20168141063e-013)

    def test_eccdot_persecond(self):
        np.testing.assert_allclose(self.sat.eccdot, -1.40011545218e-10)

    def test_mean_anomaly_rad(self):
        np.testing.assert_allclose(self.sat.mean0, 3.96130049942e0)

    def test_raandot_radpersecond(self):
        np.testing.assert_allclose(self.sat.raandot, -1.03554877709e-6)

    def test_argpdot_radpersecond(self):
        np.testing.assert_allclose(self.sat.argpdot, 7.72047261206e-7)

    def test_tle_update_r_eci_initial(self):
        np.testing.assert_allclose(self.r_eci, self.sat.r_eci)

    def test_first_visible_pass_jd(self):
        np.testing.assert_allclose(self.all_passes[0].jd, self.sat.pass_vis[0].jd)

    def test_first_visible_pass_jd_parallel_predict(self):
        np.testing.assert_allclose(self.all_passes[0].jd, self.all_passes_parallel[0].jd)

    def test_output_file(self):
        np.testing.assert_equal(filecmp.cmp(self.fcn_output_file, self.class_output_file), True)
