
import numpy as np
import pdb
from astro import geodetic, time
from kinematics import attitude

deg2rad = np.pi/180

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

def test_lla2ecef_ascension_island():
    """Example 3-2 Vallado
    """
    latgd = np.deg2rad(-7.9066357)
    lon = np.deg2rad(345.5975)
    alt = 56 / 1e3
    ecef_expected = np.array([6119.40026932, -1571.47955545, -871.56118090])
    ecef = geodetic.lla2ecef(latgd, lon, alt)
    np.testing.assert_allclose(ecef, ecef_expected)

def test_lla2ecef_washington_dc():
    lat = 38.8895 * deg2rad
    lon = -77.0353 * deg2rad
    alt = 0.3
    expected_ecef = [1115.308, -4844.546, 3982.965]
    actual_ecef = geodetic.lla2ecef(lat, lon, alt)
    np.testing.assert_allclose(actual_ecef, expected_ecef, rtol=1e-3)

def test_ecef2lla2_washington_dc():
    ecef = [1115.308, -4844.546, 3982.965]
    expected_lla =  (38.8895 * deg2rad, 282.9647 * deg2rad, 0.3)
    latgc, latgd, longd, hellp  = geodetic.ecef2lla(ecef)
    np.testing.assert_allclose((latgd, longd, hellp), expected_lla, rtol=1e-3)

def test_rv2rhoazel_vallado():
    # test example 7-1 pg.409 Vallado
    latgd = np.deg2rad(39.007)
    lon = np.deg2rad(-104.883)
    alt = 2.187
    jd = 2449857.636829

    r = np.array([-5503.79562, 62.28191, 3824.24480])
    v = np.array([-2.199987, 1.139111, 1.484966])

    expected_output = (604.68, np.deg2rad(205.6), np.deg2rad(30.7), 2.08, np.deg2rad(0.15), np.deg2rad(0.17))  
    actual_output = geodetic.rv2rhoazel(r, v, latgd, lon, alt, jd)
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-4)

def test_rv2rhoazel_vallado():
    # Example 4-1 pg. 275 Vallado
    latgd = np.deg2rad(39.007)
    lon = np.deg2rad(-104.883)
    alt = 2.19456 # kilometer
    jd, _ = time.date2jd(1994, 5, 14, 13, 11, 20.59856)

    # ECI values for position and velocity
    r_eci = np.array([1752246215, -3759563433, -1577568105])
    v_eci = np.array([-18.323, 18.332, 7.777])
    
    expected_output = (4437722626.456, 3.67834, 0.4171786,
                       -25.360829, 6.748139e-5,-2.89766e-5)
    actual_output = geodetic.rv2rhoazel(r_eci, v_eci, latgd, lon, alt, jd)
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

def test_rhoazel_equator_zenith():
    sat_eci = np.array([6378.137 + 100, 0, 0])
    site_eci = np.array([6378.137, 0, 0])
    site_lat = 0
    site_lst = 0
    
    true_rho = 100
    true_az = np.pi
    true_el = np.pi /2

    rho, az, el = geodetic.rhoazel(sat_eci, site_eci, site_lat, site_lst)
    np.testing.assert_allclose((rho, az, el), (true_rho, true_az, true_el))

def test_rhoazel_pole_zenith():
    sat_eci = np.array([0, 0, 6378.137 + 200])
    site_eci = np.array([0, 0, 6378.137])
    site_lat = np.pi / 2
    site_lst = 0
    
    true_rho = 200
    true_az = np.pi
    true_el = np.pi / 2

    rho, az, el = geodetic.rhoazel(sat_eci, site_eci, site_lat, site_lst)
    np.testing.assert_allclose((rho, az, el), (true_rho, true_az, true_el))


class TestRhoAzEl2SEZ():

    def test_zenith_stationary(self):
        rho = np.random.uniform(0, 1000)
        az = 0
        el = np.pi/2
        drho = 0
        daz = 0
        dele = 0
        rho_sez_expected = np.array([0, 0, rho])
        rho_sez, drho_sez = geodetic.rhoazel2sez(rho, az, el, drho, daz, dele)
        np.testing.assert_array_almost_equal(rho_sez, rho_sez_expected)

    def test_horizon_north(self):
        rho = np.random.uniform(0, 1000)
        az = 0
        el = 0
        drho = 0
        daz = 0
        dele = 0
        rho_sez_expected = np.array([-rho, 0, 0])
        rho_sez, drho_sez = geodetic.rhoazel2sez(rho, az, el, drho, daz, dele)
        np.testing.assert_array_almost_equal(rho_sez, rho_sez_expected)

    def test_horizon_east(self):
        rho = np.random.uniform(0, 1000)
        az = np.pi/2
        el = 0
        drho = 0
        daz = 0
        dele = 0
        rho_sez_expected = np.array([0, rho, 0])
        rho_sez, drho_sez = geodetic.rhoazel2sez(rho, az, el, drho, daz, dele)
        np.testing.assert_array_almost_equal(rho_sez, rho_sez_expected)

    def test_zenith_approaching(self):
        rho = np.random.uniform(0, 1000)
        az = 0
        el = np.pi/2
        drho = -np.random.uniform(0, 100)
        daz = 0
        dele = 0
        rho_sez_expected = np.array([0, 0, rho])
        drho_sez_expected = np.array([0, 0, drho])
        rho_sez, drho_sez = geodetic.rhoazel2sez(rho, az, el, drho, daz, dele)
        np.testing.assert_array_almost_equal(rho_sez, rho_sez_expected)
        np.testing.assert_array_almost_equal(drho_sez, drho_sez_expected)

    def test_horizon_north_receeding(self):
        rho = np.random.uniform(0, 1000)
        az = 0
        el = 0
        drho = np.random.uniform(0, 10)
        daz = 0
        dele = 0
        rho_sez_expected = np.array([-rho, 0, 0])
        drho_sez_expected = np.array([-drho, 0, 0])
        rho_sez, drho_sez = geodetic.rhoazel2sez(rho, az, el, drho, daz, dele)
        np.testing.assert_array_almost_equal(rho_sez, rho_sez_expected)
        np.testing.assert_array_almost_equal(drho_sez, drho_sez_expected)

    def test_horizon_east_approaching(self):
        rho = np.random.uniform(0, 1000)
        az = np.pi/2
        el = 0
        drho = -np.random.uniform(0, 100)
        daz = 0
        dele = 0
        rho_sez_expected = np.array([0, rho, 0])
        drho_sez_expected = np.array([0, drho, 0])
        rho_sez, drho_sez = geodetic.rhoazel2sez(rho, az, el, drho, daz, dele)
        np.testing.assert_array_almost_equal(rho_sez, rho_sez_expected)
        np.testing.assert_array_almost_equal(drho_sez, drho_sez_expected)
