"""Test the transform module
"""

from astro import transform
from kinematics.attitude import rot2, rot3
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
