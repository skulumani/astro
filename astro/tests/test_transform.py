"""Test the transform module
"""

from astro import transform, time, geodetic
from kinematics.attitude import rot2, rot3
from kinematics import attitude
import numpy as np

# TODO: Add more test functions - from book
class TestPQWtoECI():
    raan = np.deg2rad(227.89)
    inc = np.deg2rad(87.87)
    argp = np.deg2rad(53.38)

    r = np.array([6524.834, 6862.875, 6448.296])
    v = np.array([4.901320, 5.533756, -1.976341])

    def test_both_functions_equivalent(self):
        dcm_pqw2eci_vector = transform.dcm_pqw2eci_vector(self.r, self.v)
        dcm_pqw2eci_coe = transform.dcm_pqw2eci_coe(self.raan, self.inc, self.argp)
        np.testing.assert_allclose(dcm_pqw2eci_coe, dcm_pqw2eci_vector, rtol=1e-3)

class TestECEF2ENU():

    def test_equator_prime_meridian(self):
        latgd = 0
        lon = 0
        alt = 0
        dcm_ecef2enu_expected = np.array([[0, 0, 1],
                                          [1, 0, 0],
                                          [0, 1, 0]]).T
        dcm_ecef2enu = transform.dcm_ecef2enu(latgd, lon, alt)
        np.testing.assert_allclose(dcm_ecef2enu, dcm_ecef2enu_expected)
   
    def test_pole_prime_meridian(self):
        latgd = np.pi/2
        lon = 0
        alt = 0
        
        dcm_ecef2enu_expected = np.array([[0, 1, 0],
                                          [-1, 0, 0], 
                                          [0, 0, 1]])
        dcm_ecef2enu = transform.dcm_ecef2enu(latgd, lon, alt)
        np.testing.assert_array_almost_equal(dcm_ecef2enu, dcm_ecef2enu_expected)

    def test_so3(self):
        latgd = np.random.uniform(-np.pi/2, np.pi/2)
        lon = np.random.uniform(-np.pi, np.pi)
        alt = 0
        dcm = transform.dcm_ecef2enu(latgd, lon, alt)
        np.testing.assert_allclose(np.linalg.det(dcm), 1)
        np.testing.assert_array_almost_equal(dcm.T.dot(dcm), np.eye(3,3))
    
    def test_inverse_identity(self):
        latgd = np.random.uniform(-np.pi/2, np.pi/2)
        lon = np.random.uniform(-np.pi, np.pi)
        alt = 0
        dcm = transform.dcm_ecef2enu(latgd, lon, alt)
        dcm_opp = transform.dcm_enu2ecef(latgd, lon, alt)
        np.testing.assert_allclose(dcm.T, dcm_opp)

class TestECEF2SEZ():
    def test_equator_prime_meridian(self):
        latgd = 0
        lon = 0
        alt = 0
        dcm_expected = np.array([[0, 0, -1], 
                                 [0, 1, 0], 
                                 [1, 0, 0]])
        dcm = transform.dcm_ecef2sez(latgd, lon, alt)
        np.testing.assert_array_almost_equal(dcm, dcm_expected)
   
    def test_pole_prime_meridian(self):
        latgd = np.pi/2
        lon = 0
        alt = 0
        
        dcm_expected = np.array([[1, 0, 0],
                                 [0, 1, 0], 
                                 [0, 0, 1]])
        dcm = transform.dcm_ecef2sez(latgd, lon, alt)
        np.testing.assert_array_almost_equal(dcm , dcm_expected)

    def test_so3(self):
        latgd = np.random.uniform(-np.pi/2, np.pi/2)
        lon = np.random.uniform(-np.pi, np.pi)
        alt = 0
        dcm = transform.dcm_ecef2sez(latgd, lon, alt)
        np.testing.assert_allclose(np.linalg.det(dcm), 1)
        np.testing.assert_array_almost_equal(dcm.T.dot(dcm), np.eye(3,3))
    
    def test_inverse_identity(self):
        latgd = np.random.uniform(-np.pi/2, np.pi/2)
        lon = np.random.uniform(-np.pi, np.pi)
        alt = 0
        dcm = transform.dcm_ecef2sez(latgd, lon, alt)
        dcm_opp = transform.dcm_sez2ecef(latgd, lon, alt)
        np.testing.assert_allclose(dcm.T, dcm_opp)
    
    def test_dcm_loop(self):
        latitude = np.linspace(-np.pi/2, np.pi/2, 10)
        longitude = np.linspace(-np.pi, np.pi, 10)
        for lat, lon in zip(latitude, longitude):
            dcm_expected = rot2(np.pi/2 - lat, 'r').dot(rot3(lon, 'r'))
            dcm = transform.dcm_ecef2sez(lat, lon)
            np.testing.assert_allclose(dcm, dcm_expected)

class TestECEF2NED():


    def test_equator_prime_meridian(self):
        latgd = 0
        lon = 0
        alt = 0
        dcm_expected = np.array([[0, 0, 1], 
                                 [0, 1, 0], 
                                 [-1, 0, 0]])
        dcm = transform.dcm_ecef2ned(latgd, lon, alt)
        np.testing.assert_allclose(dcm, dcm_expected)
   
    def test_pole_prime_meridian(self):
        latgd = np.pi/2
        lon = 0
        alt = 0
        
        dcm_expected = np.array([[0, 1, 0],
                                 [-1, 0, 0], 
                                 [0, 0, 1]])
        dcm = transform.dcm_ecef2enu(latgd, lon, alt)
        np.testing.assert_array_almost_equal(dcm , dcm_expected)

    def test_so3(self):
        latgd = np.random.uniform(-np.pi/2, np.pi/2)
        lon = np.random.uniform(-np.pi, np.pi)
        alt = 0
        dcm = transform.dcm_ecef2ned(latgd, lon, alt)
        np.testing.assert_allclose(np.linalg.det(dcm), 1)
        np.testing.assert_array_almost_equal(dcm.T.dot(dcm), np.eye(3,3))
    
    def test_inverse_identity(self):
        latgd = np.random.uniform(-np.pi/2, np.pi/2)
        lon = np.random.uniform(-np.pi, np.pi)
        alt = 0
        dcm = transform.dcm_ecef2ned(latgd, lon, alt)
        dcm_opp = transform.dcm_ned2ecef(latgd, lon, alt)
        np.testing.assert_allclose(dcm.T, dcm_opp)

class TestPQW2LVLH():
    nu = np.random.uniform(0, 2*np.pi)
    dcm = transform.dcm_pqw2lvlh(nu)
    dcm_opp = transform.dcm_lvlh2pqw(nu)

    def test_true_anomaly_90(self):
        nu = np.deg2rad(90)
        phat = np.array([1, 0, 0])
        R_pqw2lvlh = transform.dcm_pqw2lvlh(nu)
        R_lvlh2pqw = transform.dcm_lvlh2pqw(nu)
        rhat = np.array([0, -1, 0])
        np.testing.assert_array_almost_equal(R_pqw2lvlh.dot(phat), rhat)
        np.testing.assert_array_almost_equal(R_lvlh2pqw.dot(rhat), phat)

    def test_true_anomaly_180(self):
        nu = np.deg2rad(180)
        phat = np.array([1, 0, 0])
        R_pqw2lvlh = transform.dcm_pqw2lvlh(nu)
        R_lvlh2pqw = transform.dcm_lvlh2pqw(nu)
        rhat = np.array([-1, 0, 0])
        np.testing.assert_array_almost_equal(R_pqw2lvlh.dot(phat), rhat)
        np.testing.assert_array_almost_equal(R_lvlh2pqw.dot(rhat), phat)

    def test_so3(self):
        dcm = self.dcm
        np.testing.assert_allclose(np.linalg.det(dcm), 1)
        np.testing.assert_array_almost_equal(dcm.T.dot(dcm), np.eye(3,3))
    
    def test_inverse_identity(self):
        dcm = self.dcm
        dcm_opp = self.dcm_opp
        np.testing.assert_allclose(dcm.T, dcm_opp)

class TestECI2ECEF():
    """Test to make sure we can convert a location on the earth to the correct ECEF vector
    """

    lonexp = np.deg2rad(72.5529)
    latgdexp= np.deg2rad(34.352496)
    latgcexp = np.deg2rad(34.173429)
    altexp = 5085.22 # kilometer

    eci_exp = np.array([6524.834, 6862.875, 6448.296])
    jd_exp = 2449773.0
    _, lst = time.gstlst(jd_exp, lonexp)
    eci = geodetic.site2eci(latgdexp, altexp, lst)
    ecef = geodetic.lla2ecef(latgdexp, lonexp, altexp)
    Reci2ecef = transform.dcm_eci2ecef(jd_exp)
    eci_from_ecef = Reci2ecef.T.dot(ecef)

    def test_eci_vallado(self):
        np.testing.assert_allclose(self.eci, self.eci_exp, rtol=1e-2)

    def test_eci_from_ecef(self):
        np.testing.assert_allclose(self.eci_from_ecef, self.eci_exp, rtol=1e-2)

class TestECEF2ECI():
    """Example pg.106 in BMW
    """
    lon = np.deg2rad(-57.296)
    lat = 0
    alt = 6.378 # kilometer above equator
    date = (1970, 1, 2, 6, 0, 0)
    jd, _ = time.date2jd(date[0], date[1], date[2], date[3], date[4], date[5]) 
    gst0_exp = 1.749333
    gst_exp = attitude.normalize(9.6245, 0, 2*np.pi)
    lst_exp = attitude.normalize(8.6245, 0, 2*np.pi)

    eci_exp = np.array([-0.697*6378.137, 0.718*6378.137, 0])

    gst0 = time.gsttime0(date[0])
    gst, lst = time.gstlst(jd, lon)

    eci = geodetic.site2eci(lat, alt, lst)
    ecef = geodetic.lla2ecef(lat,lon,alt)
    Reci2ecef = transform.dcm_eci2ecef(jd)
    eci_from_ecef = Reci2ecef.T.dot(ecef)

    def test_gst0(self):
        np.testing.assert_allclose(self.gst0, self.gst0_exp, rtol=1e-4)

    def test_gst(self):
        np.testing.assert_allclose(self.gst, self.gst_exp, rtol=1e-4)

    def test_lst(self):
        np.testing.assert_allclose(self.lst, self.lst_exp, rtol=1e-3)


    def test_eci(self):
        np.testing.assert_allclose(self.eci, self.eci_exp, rtol=1e-2)

    def test_eci_from_ecef(self):
        np.testing.assert_allclose(self.eci_from_ecef, self.eci_exp, rtol=1e-2)
