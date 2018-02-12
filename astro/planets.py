"""
Do lots of functions for the planets, sun and/or moon
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
import numpy as np
from kinematics import attitude
from . import kepler, constants
import pdb

twopi = 2 * np.pi
deg2rad = np.pi / 180
rad2deg = 180 / np.pi
au2km = 149597870.0

COE = namedtuple('COE', ['p', 'ecc', 'inc', 'raan', 'argp', 'nu'])


def sun_earth_eci(jd):
    """This function calculates the Geocentric Equatorial position vector for
    the Sun given the Julian Date.  This is the low precision formula and is
    valid for years from 1950 to 2050.  Accuaracy of apparent coordinates is
    0.01 degrees.  Notice many of the calculations are performed in degrees,
    and are not changed until later.  This is due to the fact that the Almanac
    uses degrees exclusively in their formulations.

    Algorithm     : Calculate the several values needed to find the vector
                    Be careful of quadrant checks

    Author        : Capt Dave Vallado  USAFA/DFAS  719-472-4109  25 Aug 1988
    In Ada        : Dr Ron Lisowski    USAFA/DFAS  719-472-4110  17 May 1995
    In MatLab     : Dr Ron Lisowski    USAFA/DFAS  719-333-4109  10 Oct 2001
    In Python     : Shankar Kulumani   GWU         630-336-6257  19 Jun 2017

    Inputs        :
    JD          - Julian Date                            days from 4713 B.C.

    Outputs       :
    RSun        - IJK Position vector of the Sun         km
    RtAsc       - Right Ascension                        rad
    Decl        - Declination                            rad

    Locals        :
    MeanLong    - Mean Longitude
    MeanAnomaly - Mean anomaly
    N           - Number of days from 1 Jan 2000
    EclpLong    - Ecliptic longitude
    Obliquity   - Mean Obliquity of the Ecliptic

    Constants     :
    Pi          -
    TwoPI       -
    InvRad      - Radians per degree

    Coupling      :

    References             :
    1996 Astronomical Almanac Pg. C24
    http://aa.usno.navy.mil/faq/docs/SunApprox.php
    """
    N = jd - 2451545.0

    meanlong = 280.461 + 0.9856474 * N
    meanlong = attitude.normalize(meanlong, 0, 360)[0]

    meananomaly = 357.528 + 0.9856003 * N
    meananomaly = attitude.normalize(meananomaly * deg2rad, 0, twopi)[0]
    if meananomaly < 0:
        meananomaly = twopi + meananomaly

    eclplong = meanlong + 1.915 * \
        np.sin(meananomaly) + 0.020 * np.sin(2 * meananomaly)
    obliquity = 23.439 - 0.0000004 * N

    meanlong = meanlong * deg2rad
    if meanlong < 0:
        meanlong = twopi + meanlong

    eclplong = eclplong * deg2rad
    obliquity = obliquity * deg2rad

    ra = np.arctan2(np.cos(obliquity) * np.sin(eclplong), np.cos(eclplong))
    dec = np.arcsin(np.sin(obliquity) * np.sin(eclplong))

    # equation of time
    eqtime = meanlong * rad2deg / 15 - ra * rad2deg / 15

    # sun vector
    sun_dist = 1.00014 - 0.01671 * \
        np.cos(meananomaly) - 0.00014 * np.cos(2 * meananomaly)

    semidiameter = 0.2666 / sun_dist  # angular semidiamter in deg
    sun_eci = [np.cos(eclplong) * sun_dist * au2km,
               np.cos(obliquity) * np.sin(eclplong) * sun_dist * au2km,
               np.sin(obliquity) * np.sin(eclplong) * sun_dist * au2km]

    return np.squeeze(sun_eci), ra, dec


def planet_coe(JD_curr, planet_flag):
    r"""Orbital elements for any of the major solar system bodies

    Given the Julian date this function will output the approximate orbital
    elements for any of the solar system bodies. It uses an analytic
    approximation for their positions and propogates the orbital elements to
    the desired date.

    Parameters
    ----------
    jd : float
        Julian date for the epoch
    planet_flag : int
        Number 0 to 8 for each of the planets (Pluto forever)

    Returns
    -------
    coe : tuple
        p : float
            Semiparameter of orbit in AU
        ecc: float
            Eccentricity of orbit
        inc : float
            Inclination in radiands wrt to ecliptic
        raan : float
            Longitude/Right ascension of teh ascending node in radians
        argp : float
            Argument of perigee in radians
        nu : float
            True anomaly in radians
    r_ecliptic : (3,) ndarray
        Position wrt J2000 mean ecliptic (Earth's orbit around the sun) in
        kilometers defined relative to the solar system barycenter
    v_ecliptic : (3,) ndarray
        Velocity wrt J2000 mean ecliptic in kilometers per second defined
        relative to the solar system barycenter
    r_icrf : (3,) ndarray
        Position wrt J2000 equatorial/ICRF in kilometers and defined relative
        to teh solar system barycenter
    v_icrf : (3,)
        Velocity wrt J2000 equatorial/ICRF in kilometers per second defined
        relative to the solar system barycenter

    See Also
    --------
    planet_approx : Actually has the data for the approximations 

    Notes
    -----
    The orbital elements are defined with respect to the mean ecliptic and
    equinox of J2000. The fundamental plane is the ecliptic, or the orbit of
    the Earth about the sun. The fundamental direction is aligned with the
    first point of Ares or the vernal equinox.

    The vectors are defined in both the ecliptic and equatorial system with
    respect to the solar system barycenter

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu   23 June 2017

    References
    ----------

    .. [1] https://ssd.jpl.nasa.gov/?planet_pos
    .. [2] MEEUS, Jean H.. Astronomical Algorithms. Willmann-Bell,
    Incorporated, 1991.

    Examples
    --------
    An example of how to use the function

    >>> planet_flag = 4
    >>> coe = planet_coe(jd, planet_flag)

    """

    # Find the JD centuries past J2000 epoch
    JD_J2000 = 2451545.0
    T = (JD_curr - JD_J2000) / 36525

    # load the elements and element rates for the planets
    (a0, adot, e0, edot, inc0, incdot, meanL0, meanLdot, lonperi0, lonperidot,
     raan0, raandot, b, c, f, s) = planet_approx(JD_curr, planet_flag)

    # compute the current elements at this JD
    a = a0 + adot * T
    ecc = e0 + edot * T
    inc = inc0 + incdot * T
    L = meanL0 + meanLdot * T
    lonperi = lonperi0 + lonperidot * T
    raan = raan0 + raandot * T

    # compute argp and M/v to complete the element set
    argp = lonperi - raan
    M = L - lonperi + b * T**2 + c * np.cos(f * T) + s * np.sin(f * T)

    # M = attitude.normalize(M, -180, 180)
    # solve kepler's equation to compute E and v
    E, nu, count = kepler.kepler_eq_E(np.deg2rad(M), ecc)
    argp = attitude.normalize(np.deg2rad(argp), 0, 2 * np.pi)[0]
    # package into a vector and output
    p = a * (1 - ecc**2)
    
    au2km = constants.au2km
    coe = COE(p=p*au2km, ecc=ecc, inc=np.deg2rad(inc), raan=np.deg2rad(raan),
              argp=argp, nu=nu)

    # convert to position and velocity vectors in J2000 ecliptic and J2000
    # Earth equatorial (ECI) reference frame
    # need to convert distances to working units of kilometers
    
    r_ecliptic, v_ecliptic, r_pqw, v_pqw = kepler.coe2rv(coe.p, coe.ecc, coe.inc,
                                                         coe.raan, coe.argp, coe.nu.item(),
                                                         constants.sun.mu)

    # convert to the Earth J2000 frame (ECI) need to rotate by the obliquity of
    # the ecliptic
    Eclip2ECI = attitude.rot1(constants.obliquity)

    r_eci = Eclip2ECI.dot(r_ecliptic)
    v_eci = Eclip2ECI.dot(v_ecliptic)

    return coe, r_ecliptic, v_ecliptic, r_eci, v_eci


def planet_approx(JD, planet_flag):
    r"""Approximate orbital elements of each of the planets

    This function provides the classical orbital elements and the rates of
    change for any of the planets in the range of 3000 BC to 3000 AD. A higher
    accuracy model if provided over the range of 1800 - 2050 AD.

    Parameters
    ----------
    jd : float
        Julian date for the epoch
    planet_flag : int
        Number 0 to 8 for each of the planets (Pluto forever)

    Returns
    -------
    a0 : float
        Semimajor axis in AU at epoch
    adot : float
        Rate of change of semimajor axis in AU/century
    e0 : float
        Eccentricty of orbit at epoch
    edot :  float
        Rate of change of eccentricy per century
    inc0 : float
        inclination in degrees at epoch
    incdot : float
        Rate of change of inclination in degrees/century
    meanL0 : float
        Mean longitude in degrees at epoch
    meanLdot : float
        Rate of change of mean longitude in degrees/century
    lonperi0 : float
        Longitude of perihelion in degrees at epoch
    lonperidot : float
        Rate of change of Longitude of perihelion in degrees/century
    raan0 : float
        Longitude of the ascending node in degrees at epoch
    raandot : float
        Rate of RAAN in degrees/century
    b, c, f, s : floats
        Additional parameters for the outer planets when using the coarse model

    See Also
    --------
    planet_coe : Use this function to compute the orbital elements

    Notes
    -----
    The orbital elements are defined with respect to the mean ecliptic and
    equinox of J2000. The fundamental plane is the ecliptic, or the orbit of
    the Earth about the sun. The fundamental direction is aligned with the
    first point of Ares or the vernal equinox.

    Rather than using this function directly, simply use planet_coe to get the
    classical orbital elements wrt J2000 ecliptic.

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu   23 June 2017

    References
    ----------

    .. [1] https://ssd.jpl.nasa.gov/?planet_pos
    .. [2] MEEUS, Jean H.. Astronomical Algorithms. Willmann-Bell,
    Incorporated, 1991.

    Examples
    --------
    An example of how to use the function


    """
    JD_1800AD = 2378497.000000
    JD_2050AD = 2469808.000000

    JD_3000BC = 625674.000000
    JD_3000AD = 2816788.000000

    if JD > JD_1800AD and JD < JD_2050AD:  # date is between 1800-2050AD and use the fine model
        b = 0.0
        c = 0.0
        f = 0.0
        s = 0.0

        if planet_flag == 0:  # mercury
            a0 = 0.38709927
            adot = 0.00000037

            e0 = 0.20563593
            edot = 0.00001906

            inc0 = 7.00497902
            incdot = -0.00594749

            meanL0 = 252.25032350
            meanLdot = 149472.67411175

            lonperi0 = 77.45779628
            lonperidot = 0.16047689

            raan0 = 48.33076593
            raandot = -0.12534081

        elif planet_flag == 1:  # venus
            a0 = 0.72333566
            adot = 0.00000390

            e0 = 0.00677672
            edot = -0.00004107

            inc0 = 3.39467605
            incdot = -0.00078890

            meanL0 = 181.97909950
            meanLdot = 58517.81538729

            lonperi0 = 131.60246718
            lonperidot = 0.00268329

            raan0 = 76.67984255
            raandot = -0.27769418
        elif planet_flag == 2:  # earth moon barycenter
            a0 = 1.00000261
            adot = 0.00000562

            e0 = 0.01671123
            edot = -0.00004392

            inc0 = -0.00001531
            incdot = -0.01294668

            meanL0 = 100.46457166
            meanLdot = 35999.37244981

            lonperi0 = 102.93768193
            lonperidot = 0.32327364

            raan0 = 0.0
            raandot = 0.0
        elif planet_flag == 3:  # mars
            a0 = 1.52371034
            adot = 0.00001847

            e0 = 0.09339410
            edot = 0.00007882

            inc0 = 1.84969142
            incdot = -0.00813131

            meanL0 = -4.55343205
            meanLdot = 19140.30268499

            lonperi0 = -23.94362959
            lonperidot = 0.44441088

            raan0 = 49.55953891
            raandot = -0.29257343
        elif planet_flag == 4:  # jupiter
            a0 = 5.20288700
            adot = -0.00011607

            e0 = 0.04838624
            edot = -0.00013253

            inc0 = 1.30439695
            incdot = -0.00183714

            meanL0 = 34.39644051
            meanLdot = 3034.74612775

            lonperi0 = 14.72847983
            lonperidot = 0.21252668

            raan0 = 100.47390909
            raandot = 0.20469106
        elif planet_flag == 5:  # saturn
            a0 = 9.53667594
            adot = -0.00125060

            e0 = 0.05386179
            edot = -0.00050991

            inc0 = 2.48599187
            incdot = 0.00193609

            meanL0 = 49.95424423
            meanLdot = 1222.49362201

            lonperi0 = 92.59887831
            lonperidot = -0.41897216

            raan0 = 113.66242448
            raandot = -0.28867794
        elif planet_flag == 6:  # uranus
            a0 = 19.18916464
            adot = -0.00196176

            e0 = 0.04725744
            edot = -0.00004397

            inc0 = 0.77263783
            incdot = -0.00242939

            meanL0 = 313.23810451
            meanLdot = 428.48202785

            lonperi0 = 170.95427630
            lonperidot = 0.40805281

            raan0 = 74.01692503
            raandot = 0.04240589
        elif planet_flag == 7:  # neptune
            a0 = 30.06992276
            adot = 0.00026291

            e0 = 0.00859048
            edot = 0.00005105

            inc0 = 1.77004347
            incdot = 0.00035372

            meanL0 = -55.12002969
            meanLdot = 218.45945325

            lonperi0 = 44.96476227
            lonperidot = -0.32241464

            raan0 = 131.78422574
            raandot = -0.00508664
        elif planet_flag == 8:  # pluto
            a0 = 39.48211675
            adot = -0.00031596

            e0 = 0.24882730
            edot = 0.00005170

            inc0 = 17.14001206
            incdot = 0.00004818

            meanL0 = 238.92903833
            meanLdot = 145.20780515

            lonperi0 = 224.06891629
            lonperidot = -0.04062942

            raan0 = 110.30393684
            raandot = -0.01183482
        else:
            print("Incorrect planet flag should be between 0 and 8")

    elif JD > JD_3000BC and JD < JD_3000AD:
        # we'll set this for the outer planets. everyone else gets zero
        b = 0.0
        c = 0.0
        f = 0.0
        s = 0.0

        if planet_flag == 0:  # mercury
            a0 = 0.38709843
            adot = 0.0

            e0 = 0.20563661
            edot = 0.00002123

            inc0 = 7.00559432
            incdot = -0.00590158

            meanL0 = 252.25166724
            meanLdot = 149472.67486623

            lonperi0 = 77.45771895
            lonperidot = 0.15940013

            raan0 = 48.33961819
            raandot = -0.12214182

        elif planet_flag == 1:  # venus
            a0 = 0.72332102
            adot = -0.00000026

            e0 = 0.00676399
            edot = -0.00005107

            inc0 = 3.39777545
            incdot = 0.00043494

            meanL0 = 181.97970850
            meanLdot = 58517.81560260

            lonperi0 = 131.76755713
            lonperidot = 0.05679648

            raan0 = 76.67261496
            raandot = -0.27274174

        elif planet_flag == 2:  # earth moon barycenter
            a0 = 1.00000018
            adot = -0.00000003

            e0 = 0.01673163
            edot = -0.00003661

            inc0 = -0.00054346
            incdot = -0.01337178

            meanL0 = 100.46691572
            meanLdot = 35999.37306329

            lonperi0 = 102.93005885
            lonperidot = 0.31795260

            raan0 = -5.11260389
            raandot = -0.24123856
        elif planet_flag == 3:  # mars
            a0 = 1.52371243
            adot = 0.00000097

            e0 = 0.09336511
            edot = 0.00009149

            inc0 = 1.85181869
            incdot = -0.00724757

            meanL0 = -4.56813164
            meanLdot = 19140.29934243

            lonperi0 = -23.91744784
            lonperidot = 0.45223625

            raan0 = 49.71320984
            raandot = -0.26852431
        elif planet_flag == 4:  # jupiter
            a0 = 5.20248019
            adot = -0.00002864

            e0 = 0.04853590
            edot = 0.00018026

            inc0 = 1.29861416
            incdot = -0.00322699

            meanL0 = 34.33479152
            meanLdot = 3034.90371757

            lonperi0 = 14.27495244
            lonperidot = 0.18199196

            raan0 = 100.29282654
            raandot = 0.13024619
        elif planet_flag == 5:  # saturn
            a0 = 9.54149883
            adot = -0.00003065

            e0 = 0.05550825
            edot = -0.00032044

            inc0 = 2.49424102
            incdot = 0.00451969

            meanL0 = 50.07571329
            meanLdot = 1222.11494724

            lonperi0 = 92.86136063
            lonperidot = 0.54179478

            raan0 = 113.63998702
            raandot = -0.25015002
        elif planet_flag == 6:  # uranus
            a0 = 19.18797948
            adot = -0.00020455

            e0 = 0.04685740
            edot = -0.00001550

            inc0 = 0.77298127
            incdot = -0.00180155

            meanL0 = 314.20276625
            meanLdot = 428.49512595

            lonperi0 = 172.43404441
            lonperidot = 0.09266985

            raan0 = 73.96250215
            raandot = 0.05739699
        elif planet_flag == 7:  # neptune
            a0 = 30.06952752
            adot = 0.00006447

            e0 = 0.00895439
            edot = 0.00000818

            inc0 = 1.77005520
            incdot = 0.00022400

            meanL0 = 304.22289287
            meanLdot = 218.46515314

            lonperi0 = 46.68158724
            lonperidot = 0.01009938

            raan0 = 131.78635853
            raandot = -0.00606302
        elif planet_flag == 8:  # pluto
            a0 = 39.48686035
            adot = 0.00449751

            e0 = 0.24885238
            edot = 0.00006016

            inc0 = 17.14104260
            incdot = 0.00000501

            meanL0 = 238.96535011
            meanLdot = 145.18042903

            lonperi0 = 224.09702598
            lonperidot = -0.00968827

            raan0 = 110.30167986
            raandot = -0.00809981
        else:
            print("Incorrect planet flag should be between 0 and 8")
    else:
        raise ValueError('Julian Date is outside of the range 3000BC-3000AD')

    return (a0, adot, e0, edot, inc0, incdot, meanL0, meanLdot, lonperi0,
            lonperidot, raan0, raandot, b, c, f, s)


def asteroid_epoch(ast_flag):
    """
    This holds the orbital elements for the asteroids as taken from JPL

    Return the state at the JD epoch for use in a propogate function
    """

    if ast_flag == 0:
        # 2008 EV5
        a = 0.9582899238313918
        ecc = 0.08348599378460778
        p = a * (1 - ecc**2)
        inc = np.deg2rad(7.436787362690259)
        raan = np.deg2rad(93.39122898787916)
        argp = np.deg2rad(234.8245876826614)
        M = np.deg2rad(3.409187469072454)
        JD_epoch = 2457800.5
    elif ast_flag == 1:
        # Itokawa
        a = 1.324163617639197
        ecc = 0.28011765678781
        p = a * (1 - ecc**2)
        inc = np.deg2rad(1.62145641293925)
        raan = np.deg2rad(69.07992986350325)
        argp = np.deg2rad(162.8034822691509)
        M = np.deg2rad(131.4340297670125)
        JD_epoch = 2457800.5
    elif ast_flag == 2:
        # Bennu
        a = 1.126391026007489
        ecc = 0.2037451112033579
        p = a * (1 - ecc**2)
        inc = np.deg2rad(6.034939195483961)
        raan = np.deg2rad(2.060867837066797)
        argp = np.deg2rad(66.22306857848962)
        M = np.deg2rad(101.7039476994243)
        JD_epoch = 2455562.5
    else:
        print("No such asteroid defined yet. Using Itokawa so nothing breaks instead")

        # Itokawa
        a = 1.324163617639197
        ecc = 0.28011765678781
        p = a * (1 - ecc**2)
        inc = np.deg2rad(1.62145641293925)
        raan = np.deg2rad(69.07992986350325)
        argp = np.deg2rad(162.8034822691509)
        M = np.deg2rad(131.4340297670125)
        JD_epoch = 2457800.5

    # convert M to nu for later and output the coe
    E, nu, count = kepler_eq_E(M, ecc)

    epoch = (p, ecc, inc, raan, argp, attitude.normalize(nu, 0, 2 * np.pi), JD_epoch)

    return epoch


def asteroid_coe(JD_curr, ast_flag):
    """
        Output the current COE for the chosen asteroid
    """

    # load the asteroid COE at the epoch
    (p, ecc, inc, raan, argp, nu_0, JD_epoch) = asteroid_epoch(ast_flag)

    # compute the delta t
    delta_t = (JD_curr - JD_epoch) * 86400
    mu = 1.32712440018e20  # m^3 / s^2
    mu = 1 / 149597870700**3 * mu  # au^3 / sec^2

    # propogate to the current JD_curr
    (E_f, M_f, nu_f) = tof_delta_t(p, ecc, mu, nu_0, delta_t)

    # output the current COE
    coe = (p, ecc, inc, raan, argp, attitude.normalize(nu_f, 0, 2 * np.pi))

    return coe
