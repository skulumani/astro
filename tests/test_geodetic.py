
import numpy as np

from .. import geodetic

def test_equatorial_prime_meridian_lla2ecef():
    lat = 0 * np.pi/180
    lon = 0 * np.pi/180
    alt = 0
    expected_ecef = np.array([6378.137, 0, 0])
    actual_ecef = geodetic.lla2ecef(lat, lon, alt)
    np.testing.assert_array_almost_equal(actual_ecef, expected_ecef)

def test_equatorial_90_west_lla2ecef():
    lat = 0 * np.pi/180
    lon = 90 * np.pi/180
    alt = 0
    expected_ecef = np.array([0, 6378.137, 0])
    actual_ecef = geodetic.lla2ecef(lat, lon, alt)
    np.testing.assert_array_almost_equal(actual_ecef, expected_ecef)

def test_north_pole_lla2eccef():
    lat = 90 * np.pi/180
    lon = 0 * np.pi/180
    alt = 0
    expected_ecef = np.array([0, 0, 6356.7523142451792])
    actual_ecef = geodetic.lla2ecef(lat, lon, alt)
    np.testing.assert_array_almost_equal(actual_ecef, expected_ecef)

