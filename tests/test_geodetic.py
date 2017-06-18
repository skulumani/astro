
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

def test_lla2ecef_vallado():
    """Example 3-3 from Vallado
    """
    latgd = 34.352496 * np.pi/180
    lon = 46.4464 * np.pi/180
    hellp = 5085.22 # km
    expected_ecef = [6524.834, 6862.875, 6448.296] # km
    actual_ecef = geodetic.lla2ecef(latgd, lon, hellp)
    np.testing.assert_array_almost_equal(actual_ecef, expected_ecef, decimal=2)

def test_ecef2lla_vallado():
    """Example 3-3 from Vallado
    """
    ecef = [6524.834, 6862.875, 6448.296]
    # this is in latgd, lon, hellp in km
    expected_lla = [34.352496*np.pi/180, 46.4464 * np.pi/180, 5085.22]
    latgc, latgd, lon, hellp = geodetic.ecef2lla(ecef) 
    np.testing.assert_allclose([latgd, lon, hellp], expected_lla, rtol=1e-3)

def test_gc2gd_vallado():
    latgc = 34.173429 * np.pi/180
    expected_latgd = 34.352496 *np.pi/180
    actual_latgd = geodetic.gc2gd(latgc)
    np.testing.assert_allclose(actual_latgd, expected_latgd)

def test_gd2gc_vallado():
    latgd = 34.352496 * np.pi/180
    expected_latgc = 34.173429 * np.pi/180
    actual_latgc = geodetic.gd2gc(latgd)
    np.testing.assert_allclose(actual_latgc, expected_latgc)

def test_lla2ecef_usafa():
    lat = 39.006 * np.pi/180
    lon = -104.883 * np.pi/180
    alt = 2.184
    expected_ecef = [-1275.139, -4798.054, 3994.209]
    actual_ecef = geodetic.lla2ecef(lat, lon, alt)
    np.testing.assert_allclose(actual_ecef, expected_ecef, rtol=1e-3)

