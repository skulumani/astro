"""Keplerian method to do astrodynamics
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from kinematics import attitude
import pdb


def coe2rv(p_in, ecc_in, inc_in, raan_in, arg_p_in, nu_in, mu):
    """
       Purpose:
           - Convert the classical orbital elements (COEs) to position (R)
               and velocity (V) vectors.

       [R_ijk,V_ijk,R_pqw,V_pqw] =
       coe2rv(p,ecc,inc,raan,arg_p,nu,mu,arglat,truelon,lonper )

       Inputs:
           - p - semi-major axis (km)
           - ecc - eccentricity
           - raan - right acsension of the ascending node (rad) 0 < raan <2*pi
           - inc - inclination (rad) 0 < inc < pi
           - arg_p - argument of periapsis (rad) 0 < arg_p < 2*pi
           - nu - true anomaly (rad) 0 < nu < 2*pi
           - mu - gravitational parameter of central body (km^3/sec^2).
           - arglat - argument of latitude(CI) rad 0 <arglat < 2*pi
           - truelon - true longitude (CE) rad 0 < truelon < 2*pi
           - lonper - longitude of periapsis rad 0 < lonper < 2*pi

       Outpus:
           - R_ijk - position vector in inertial frame (km)
           - V_ijk - velocity vector in inertial frame (km/sec)
           - R_pqw - position vecotr in perifocal frame (km)
           - V_pqw - velocity vector in perifocal frame (km/sec)

       Dependencies:
           - ROT1 - elementary rotation about first axis
           - ROT2 - elementary rotation about second axis
           - ROT3 - elementary rotation about third axis

       Author:
           - Shankar Kulumani 18 Aug 2012
               - revised from code written at USAFA Fall 2007
           - Shankar Kulumani 18 Sept 2012
               - modified rotation matrix from PQW to ECI
           - Shankar Kulumani 29 Sept 2012
               - modified input to take semi-latus rectum to allow calculation
               for all orbit types
           - Shankar Kulumani 7 Dec 2014
               - loop for vector inputs
           - Shankar Kulumani 7 Dec 2016
               - Convert to python and remove vectorized inputs

    """

    tol = 1e-9
    # check if array inputs
    if not hasattr(p_in, "__iter__"):
        p_in = np.asarray([p_in], dtype=np.float)
        ecc_in = np.asarray([ecc_in], dtype=np.float)
        inc_in = np.asarray([inc_in], dtype=np.float)
        raan_in = np.asarray([raan_in], dtype=np.float)
        arg_p_in = np.asarray([arg_p_in], dtype=np.float)
        nu_in = np.asarray([nu_in], dtype=np.float)
    # make sure all inputs are the same size
    if not (p_in.shape == ecc_in.shape == inc_in.shape == raan_in.shape ==
            arg_p_in.shape == nu_in.shape):
        print("All inputs must be the same shape")
        print("p: {}".format(p_in.shape))
        print("ecc: {}".format(ecc_in.shape))
        print("inc: {}".format(inc_in.shape))
        print("raan: {}".format(raan_in.shape))
        print("argp: {}".format(arg_p_in.shape))
        print("nu: {}".format(nu_in.shape))

    r_pqw_out = []
    v_pqw_out = []
    r_ijk_out = []
    v_ijk_out = []

    for p, ecc, inc, raan, arg_p, nu in zip(p_in, ecc_in, inc_in, raan_in,
            arg_p_in, nu_in):
        # check eccentricity for special cases
        if (ecc < tol):
            # circular equatorial
            if (inc < tol) or (abs(inc - np.pi) < tol):
                arg_p = 0.0
                raan = 0.0
                # nu   = truelon
                nu = raan + arg_p + nu
            else:
                # circular inclined
                arg_p = 0.0
                nu = arg_p + nu
                # nu = arglat
        elif ((inc < tol) or (abs(inc - np.pi) < tol)):    # elliptical equatorial
            arg_p = raan + arg_p
            raan = 0.0
            # arg_p=lonper

        cosnu = np.cos(nu)
        sinnu = np.sin(nu)
        radius = p / (1 + ecc * cosnu)
        # semi latus rectum check
        if (abs(p) < 0.0001):
            p = 0.0001

        # calculate postion and velocity in perifocal frame
        r_pqw = radius * np.array([cosnu, sinnu, 0])
        v_pqw = np.sqrt(mu / p) * np.array([-sinnu, (ecc + cosnu), 0])

        PI = attitude.rot3(raan).dot(
            attitude.rot1(inc)).dot(attitude.rot3(arg_p))

        # rotate postion and velocity vectors to inertial frame
        r_ijk = np.dot(PI, r_pqw)
        v_ijk = np.dot(PI, v_pqw)

        r_pqw_out.append(r_pqw)
        v_pqw_out.append(v_pqw)
        r_ijk_out.append(r_ijk)
        v_ijk_out.append(v_ijk)

    return (np.squeeze(r_ijk_out), np.squeeze(v_ijk_out), np.squeeze(r_pqw_out),
            np.squeeze(v_pqw_out))


def kepler_eq_E(M_in, ecc_in):
    """
    (E,nu,count) = kepler_eq_E(M,ecc)
    Purpose:
       - This function solves Kepler's equation for eccentric anomaly
       given a mean anomaly using a newton-rapson method.
           - Will work for elliptical/parabolic/hyperbolic orbits

    Inputs:
       - M - mean anomaly in rad -2*pi < M < 2*pi
       - ecc - eccentricity 0 < ecc < inf

    Outputs:
       - E - eccentric anomaly in rad 0 < E < 2*pi
       - nu - true anomaly in rad 0 < nu < 2*pi

    Dependencies:
       - none

    Author:
       - Shankar Kulumani 15 Sept 2012
           - rewritten from code from USAFA
           - solve for elliptical orbits add others later
       - Shankar Kulumani 29 Sept 2012
           - added parabolic/hyperbolic functionality
       - Shankar Kulumani 7 Dec 2014
          - added loop for vector inputs
       - Shankar Kulumani 2 Dec 2016
          - converted to python and removed the vector inputs

    References
       - USAFA Astro 321 LSN 24-25
       - Vallado 3rd Ed pg 72
    """
    tol = 1e-9
    max_iter = 50

    E_out = []
    nu_out = []
    count_out = []

    if not hasattr(M_in, "__iter__"):
        M_in = [M_in]

    if not hasattr(ecc_in, "__iter__"):
        ecc_in = [ecc_in]

    for M, ecc in zip(M_in, ecc_in):
        # eccentricity check
        """
            HYPERBOLIC ORBIT
        """
        if ecc - 1.0 > tol:  # eccentricity logic
            # initial guess
            if ecc < 1.6:  # initial guess logic
                if M < 0.0 and (M > -np.pi or M > np.pi):
                    E_0 = M - ecc
                else:
                    E_0 = M + ecc

            else:
                if ecc < 3.6 and np.absolute(M) > np.pi:
                    E_0 = M - np.sign(M) * ecc;
                else:
                    E_0 = M / (ecc - 1.0);

            # netwon's method iteration to find hyperbolic anomaly
            count = 1
            E_1 = E_0 + ((M - ecc * np.sinh(E_0) + E_0) /
                         (ecc * np.cosh(E_0) - 1.0))
            while ((np.absolute(E_1 - E_0) > tol) and (count <= max_iter)):
                E_0 = E_1
                E_1 = E_0 + ((M - ecc * np.sinh(E_0) + E_0) /
                             (ecc * np.cosh(E_0) - 1.0))
                count = count + 1

            E = E_0
            # find true anomaly
            sinv = -(np.sqrt(ecc * ecc - 1.0) * np.sinh(E_1)) / \
                     (1.0 - ecc * np.cosh(E_1))
            cosv = (np.cosh(E_1) - ecc) / (1.0 - ecc * np.cosh(E_1))
            nu = np.arctan2(sinv, cosv);
        else:
            """
                PARABOLIC
            """
            if np.absolute(ecc - 1.0) < tol:  # parabolic logic
                count = 1

                S = 0.5 * (np.pi / 2 - np.arctan(1.5 * M))
                W = np.arctan(np.tan(S)**(1.0 / 3.0))

                E = 2.0 * 1.0 / np.tan(2.0 * W)

                nu = 2.0 * np.arctan(E)
            else:
                """
                    ELLIPTICAl
                """
                if ecc > tol:   # elliptical logic

                    # determine intial guess for iteration
                    if M > -np.pi and (M < 0 or M > np.pi):
                        E_0 = M - ecc
                    else:
                        E_0 = M + ecc

                    # newton's method iteration to find eccentric anomaly
                    count = 1
                    E_1 = E_0 + (M - E_0 + ecc * np.sin(E_0)) / \
                                 (1.0 - ecc * np.cos(E_0))
                    while ((np.absolute(E_1 - E_0) > tol) and (count <= max_iter)):
                        count = count + 1
                        E_0 = E_1
                        E_1 = E_0 + (M - E_0 + ecc * np.sin(E_0)) / \
                                     (1.0 - ecc * np.cos(E_0))

                    E = E_0

                    # find true anomaly
                    sinv = (np.sqrt(1.0 - ecc * ecc) * np.sin(E_1)) / \
                            (1.0 - ecc * np.cos(E_1))
                    cosv = (np.cos(E_1) - ecc) / (1.0 - ecc * np.cos(E_1))
                    nu = np.arctan2(sinv, cosv)
                else:
                    """
                        CIRCULAR
                    """
                    # -------------------- circular -------------------
                    count = 0
                    nu = M
                    E = M

        E_out.append(E)
        nu_out.append(attitude.normalize(nu, 0, 2 * np.pi))
        count_out.append(count)

    return (np.squeeze(E_out), np.squeeze(nu_out), np.squeeze(count_out))


def conic_orbit(p, ecc, inc, raan, arg_p, nu_i, nu_f):
    """Plot conic orbit

        Purpose:
           - Uses the polar conic equation to plot a conic orbit

        [x y z xs ys zs ] = conic_orbit(p,ecc,inc,raan,arg_p,nu_i,nu_f)

        Inputs:
           - p - semi-major axis (km)
           - ecc - eccentricity
           - raan - right acsension of the ascending node (rad) 0 < raan <
           2*pi
           - inc - inclination (rad) 0 < inc < pi
           - arg_p - argument of periapsis (rad) 0 < arg_p < 2*pi
           - nu_i - initial true anomaly (rad) 0 < nu < 2*pi
           - nu_f - final true anomaly (rad) 0 < nu < 2*pi

        Outputs:
           - none

        Dependencies:
           - ROT1,ROT2,ROT3 - principle axis rotation matrices

        Author:
           - Shankar Kulumani 1 Dec 2012
               - list revisions
           - Shankar Kulumani 5 Dec 2014
               - added outputs for orbit gui functions

        References
           - AAE532
    """

    tol = 1e-9
    step = 1000

    # v = true anomaly
    if nu_f > nu_i:
        v = np.linspace(nu_i, nu_f, step)
    else:
        v = np.linspace(nu_i, nu_f + 2 * np.pi, step)

    if ecc - 1 > tol:  # hyperbolic
        turn_angle = np.acos(-1.0 / ecc)
        v = np.linespace(-turn_angle, turn_angle, step);

        if nu_i > pi:
            nu_i = nu_i - 2 * np.pi

        r = p / (1 + ecc * np.cos(v))
        rs = p / (1 + ecc * np.cos(nu_i))

    elif np.absolute(ecc - 1) < tol:  # parabolic
        v = np.linspace(-np.pi, np.pi, step);
        if nu_i > np.pi:
            nu_i = nu_i - 2 * np.pi

        r = p / 2 * (1 + np.tan(v / 2)**2);
        rs = p / 2 * (1 + np.tan(nu_i / 2)**2);
    else:
        # conic equation for elliptical orbit
        r = p / (1 + ecc * np.cos(v));
        rs = p / (1 + ecc * np.cos(nu_i));

    x = r * np.cos(v)
    y = r * np.sin(v)
    z = np.zeros_like(x)

    xs = rs * np.cos(nu_i)
    ys = rs * np.sin(nu_i)
    zs = 0
    # rotate orbit plane to correct orientation

     # M_rot = [cos(raan) * cos(arg_p) - sin(raan) * cos(inc) * sin(arg_p) -cos(raan) * sin(arg_p) - sin(raan) * cos(inc) * cos(arg_p) sin(raan) * sin(inc);
     #         sin(raan) * cos(arg_p) + cos(raan) * cos(inc) * sin(arg_p) -sin(raan) * sin(arg_p) + cos(raan) * cos(inc) * cos(arg_p) -cos(raan) * sin(inc);
     #         sin(inc) * sin(arg_p) sin(inc) * cos(arg_p) cos(inc);];
    dcm_pqw2eci = np.dot(
        np.dot(attitude.ROT3(-raan), attitude.ROT1(-inc)), attitude.ROT3(-arg_p));

    orbit_plane = np.dot(dcm_pqw2eci, np.array([x, y, z]));

    x = orbit_plane[0, :]
    y = orbit_plane[1, :]
    z = orbit_plane[2, :]

    sat_pos = np.dot(dcm_pqw2eci, np.array([xs, ys, zs]));

    xs = sat_pos[0]
    ys = sat_pos[1]
    zs = sat_pos[2]

    return (x, y, z, xs, ys, zs)


def nu2anom(nu, ecc):
    """
    [E M] = ecc_anomaly(nu,ecc)

       Purpose:
           - Calculates the eccentric and mean anomaly given eccentricity and
           true anomaly

       Inputs:
           - nu - true anomaly in rad -2*pi < nu < 2*pi
           - ecc - eccentricity of orbit 0 < ecc < inf

       Outputs:
           - E - (elliptical/parabolic/hyperbolic) eccentric anomaly in rad
               0 < E < 2*pi
           - M - mean anomaly in rad 0 < M < 2*pi

       Dependencies:
           - none

       Author:
           - Shankar Kulumani 5 Dec 2016
                - Convert to python
           - Shankar Kulumani 15 Sept 2012
               - modified from USAFA code and notes from AAE532
               - only elliptical case will add other later
           - Shankar Kulumani 17 Sept 2012
               - added rev check to reduce angle btwn 0 and 2*pi

       References
           - AAE532 notes
           - Vallado 3rd Ed
    """

    small = 1e-9

    if ecc <= small:  # circular
         E = nu
         M = nu
    elif small < ecc and ecc <= 1 - small:  # elliptical
        sine = (np.sqrt(1.0 - ecc * ecc) * np.sin(nu)) / \
                (1.0 + ecc * np.cos(nu))
        cose = (ecc + np.cos(nu)) / (1.0 + ecc * np.cos(nu))

        E = np.arctan2(sine, cose)
        M = E - ecc * np.sin(E)

        E = attitude.normalize(E, 0, 2 * np.pi)
        M = attitude.normalize(M, 0, 2 * np.pi)

    elif np.absolute(ecc - 1) <= small:  # parabolic
        B = np.tan(nu / 2)

        E = B
        M = B + 1.0 / 3 * B**3

        # E = revcheck(E);
        # M = revcheck(M);
    elif ecc > 1 + small:  # hyperbolic
            sine = (np.sqrt(ecc**2 - 1) * np.sin(nu)) / \
                    (1.0 + ecc * np.cos(nu))
            H = np.arcsinh(sine)
            E = H
            M = ecc * np.sinh(H) - H

            # E = revcheck(E);
            # M = revcheck(M);
    else:
        print("Eccentricity is out of bounds 0 < ecc < inf")

    return (E, M)


def tof_delta_t(p, ecc, mu, nu_0, delta_t):
    """
        Propogate a COE into the future
    """

    tol = 1e-9
    # calculate initial eccentric anomaly and mean anomaly
    E_0, M_0 = nu2anom(nu_0, ecc)

    # check eccentricity
    if np.absolute(ecc - 1) < tol:  # parabolic
        n = 2 * np.sqrt(mu / p**3)
    else:
        a = p / (1 - ecc**2)
        # calculate mean motion
        n = np.sqrt(mu / a**3)

    # calculate mean anomaly after delta t

    M_f = M_0 + n * delta_t
    k = np.floor(M_f / (2 * np.pi))
    M_f = M_f - 2 * np.pi * k
    # calculate eccentric anomaly from mean anomaly (newton iteration)

    E_f, nu_f, count = kepler_eq_E(M_f, ecc)

    return (E_f, M_f, nu_f)


def elp_orbit_el(p, ecc, inc, raan, arg_p, nu, mu):
    """Elliptical Orbit Characteristics/Elements

    Purpose:
        - Calculates elliptical orbital parameters using conic equations

    Inputs:
        - p - semi-major axis (km)
        - ecc - eccentricity
        - raan - right acsension of the ascending node (rad) 0 < raan <
        2*pi
        - inc - inclination (rad) 0 < inc < pi
        - arg_p - argument of periapsis (rad) 0 < arg_p < 2*pi
        - nu - true anomaly (rad) 0 < nu < 2*pi
        - mu - gravitational parameter of central body (km^3/sec^2).
        - arglat - argument of latitude(CI) rad 0 <arglat < 2*pi
        - truelon - true longitude (CE) rad 0 < truelon < 2*pi
        - lonper - longitude of periapsis rad 0 < lonper < 2*pi

    Outputs:
        - a - semi-major axis in km
        - h - magnitude of angular momentum vector in km^2/sec
        - period - period of orbit in seconds
        - sme - specific mechanical energy of orbit in km^2/sec^2
        - r_per - radius of periapsis in km
        - r_apo - radius of apoapsis in km
        - r - current radius in km
        - v - current velocity in km/sec
        - v_circ - equivalent circular orbit velocity at current radius in
        km/sec
        - v_esc - escape speed at current radius km/sec
        - fpa - flight path angle in rad 0 < fpa < pi/2
        - E - eccentric anomaly in rad 0 < E < 2*pi
        - M - mean anomaly in rad 0 < M < 2*pi
        - n - mean motion in 1/sec

    Dependencies:
        - fpa_solve - find flight path angle
        - ecc_anomaly - find eccentric and mean anomaly given true
        anomaly
        - aae532_constants - AAE532 class constants
        - ROT3 - simple rotation about third axis
        - use orbit_el as driver function to print results to screen

    Author:
        - Shankar Kulumani 16 Sept 2012
            - used code from AAE532 PS4
        - Shankar Kulumani 19 Sept 2012
            - added escape speed
            - added print capability
            - added constants
            - added mean motion
            - added lvlh and pqw vectors
        - Shankar Kulumani 28 June 2017
            - Convert to Python for MAE3145 class

    References
        - AAE532
        - Any astrodynamics book/internet
    """
    # calculate semi-major axis
    a = p / (1 - np.ecc**2)  # km
    # angular momentum scalar
    h = np.sqrt(p * mu)  # km^2/sec
    # period of orbit
    period = 2 * np.pi * np.sqrt(a**3 / mu)  # sec
    # specific mechanical energy
    sme = -mu / (2 * a)  # km^2/sec^2

    fpa = fpa_solve(nu, ecc)  # radians

    # radius of periapsis and apoapsis
    r_per = a * (1 - ecc)  # km
    r_apo = a * (1 + ecc)  # km

    # convert to pqw (perifocal frame)
    r_ijk, v_ijk, r_pqw, v_pqw = coe2rv(p, ecc, inc, raan, arg_p, nu, mu)

    r = np.linalg.norm(r_pqw)
    v = np.linalg.norm(v_pqw)

    # convert position and velocity to lvlh frame
    r_lvlh = np.array([r, 0, 0])  # [r_hat theta_hat h_hat] km
    # [r_hat theta_hat h_hat] km/sec
    v_lvlh = v * np.array([np.sin(fpa), np.cos(fpa), 0])

    v_circ = np.sqrt(mu / r)

    v_esc = np.sqrt(2) * v_circ

    E, M = nu2anom(nu, ecc)  # rad

    n = np.sqrt(mu / a**3)  # mean motion in 1/sec

    return (a, h, period, sme, fpa, r_per, r_apo, r_ijk, v_ijk, r_pqw, v_pqw,
            r_lvlh, v_lvlh, r, v, v_circ, v_esc, E, M, n)
