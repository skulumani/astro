"""test all the functions in planets 
"""
import numpy as np

from astro import planets, time, constants
import pdb


rtol = 1e-0

def test_sun_earth_eci_usafa():
    jd = 2453905.50000000
    expected_rsun = [6371243.918400, 139340081.269385, 60407749.811252]
    rsun, ra, dec = planets.sun_earth_eci(jd)
    np.testing.assert_allclose(rsun, expected_rsun, rtol=1e-4)

def test_sun_eci_vallado():
    jd, mjd = time.date2jd(1994, 4, 2, 0, 0, 0)
    jd_vallado = 2449444.5
    rsun_vallado = np.array([146241097,
                             28574940,
                             12389196])
    ra_vallado, dec_vallado = np.deg2rad(11.056071), np.deg2rad(4.7529393)
    rsun, ra, dec = planets.sun_earth_eci(jd)


    np.testing.assert_allclose(jd, jd_vallado)
    np.testing.assert_allclose(rsun, rsun_vallado, rtol=1e-4)
    np.testing.assert_allclose((ra, dec), (ra_vallado, dec_vallado), rtol=1e-4)

class TestMercuryCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 0)
    
    eccexp = 2.056334003441381e-01
    incexp = 7.003938518641813e+00
    raanexp = 4.830876318589697e+01
    argpexp = 2.917536389390346e+01
    nuexp = 2.786392215752991e+01
    pexp = 3.870982978406551e-1 * (1 - eccexp**2) * constants.au2km

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)

class TestVenusCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 1)
    
    eccexp = 6.792032469514713e-03
    incexp = 3.394477356853931e+00
    raanexp = 7.662882872939052e+01
    argpexp = 5.473647539038130e+01
    nuexp = 1.975782148111063e+02
    pexp = 7.233262662615381e-01 * (1 - eccexp**2) * constants.au2km 

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)


class TestEarthCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 2)
    
    eccexp =1.579421783270573e-02
    incexp = 3.001710480218142e-03
    raanexp = 1.482633325004871e+02
    argpexp = 3.153399510406209e+02
    nuexp = 1.688484428230512e+02
    pexp = 1.000936277866668e+00 * (1 - eccexp**2) * constants.au2km

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(-np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)

class TestMarsCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 3)
    
    eccexp = 9.352430073957714e-02
    incexp = 1.848364578438978e+00
    raanexp = 4.950693065746390e+01
    argpexp = 2.866247286888100e+02
    nuexp = 1.327364669246639e+02
    pexp = 1.523603073362533e+00 * (1 - eccexp**2)  * constants.au2km

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)

class TestJupiterCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 4)
    
    eccexp = 4.886730265850301e-02
    incexp =  1.303742758808121e+00
    raanexp = 1.005110590888574e+02
    argpexp = 2.737359701611508e+02
    nuexp = 1.895852218694279e+02
    pexp = 5.202296207914233e+00 * (1 - eccexp**2)  * constants.au2km

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)

class TestSaturnCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 5)
    
    eccexp = 5.196085002126025e-02
    incexp = 2.487587507263742e+00
    raanexp = 1.135815401076222e+02
    argpexp = 3.402669207862293e+02
    nuexp = 1.706484255046402e+02
    pexp = 9.568736466329314e+00 * (1 - eccexp**2)  * constants.au2km

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)

class TestUranusCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 6)
    
    eccexp = 4.964052992824323e-02
    incexp = 7.701258737658703e-01
    raanexp = 7.413417028463512e+01
    argpexp = 9.896398369052795e+01
    nuexp = 2.118953182102191e+02
    pexp = 1.913027860362278e+01 * (1 - eccexp**2)  * constants.au2km

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)

class TestNeptuneCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 7)
    
    eccexp = 6.564105141096746e-03
    incexp = 1.766501817898360e+00
    raanexp = 1.317099904241699e+02
    argpexp = 2.795450061842975e+02
    nuexp = 2.909187977999030e+02
    pexp = 3.001973084229905e+01 * (1 - eccexp**2)  * constants.au2km

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)

class TestPlutoCOE():
    jd = 2457928.5
    coe, _, _, _, _ = planets.planet_coe(jd, 8)
    
    eccexp = 2.496793714912267e-01
    incexp = 1.711614987378614e+01
    raanexp = 1.102980595626739e+02
    argpexp = 1.121586170630204e+02
    nuexp = 6.522471039312354e+01
    pexp = 3.929274514682939e+01 * (1 - eccexp**2)  * constants.au2km

    def test_p(self):
        np.testing.assert_allclose(self.coe.p, self.pexp, rtol=rtol)

    def test_ecc(self):
        np.testing.assert_allclose(self.coe.ecc, self.eccexp, rtol=rtol)

    def test_inc(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.inc), self.incexp, rtol=rtol)

    def test_raan(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.raan), self.raanexp, rtol=rtol)

    def test_argp(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.argp), self.argpexp, rtol=rtol)

    def test_nu(self):
        np.testing.assert_allclose(np.rad2deg(self.coe.nu), self.nuexp, rtol=rtol)
