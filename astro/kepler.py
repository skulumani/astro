"""Keplerian method to do astrodynamics
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from kinematics import attitude
import pdb
from . import constants
import sys

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


def perapo2aecc(r_per, r_apo):
    """Apoapsis/Periapsis to Semi-major axis and Eccentricity

     a, p, ecc = perapo2aecc(r_per,r_apo)

    Inputs: 
    - r_per - periapsis distance in km
    - r_apo - apoapsis distance in km

    Outputs: 
    - a - semi-major axis in km
    - ecc - eccentricity

    Dependencies: 
    - none

    Author: 
    - Shankar Kulumani 19 Sept 2012
    - list revisions
    - Shankar Kulumani 31 Oct 2012
    - added semi-latus rectum
    - Shankar Kulumani 2 Oct 2017
        - convert to Python

    References
    - AAE532 Notes/PS5
    """

    ecc = (r_apo - r_per) / (r_per + r_apo)
    a = r_per / (1 - ecc)

    p = a * (1 - ecc**2)

    return a, p, ecc


def hne_vec(r, v, mu):
    r"""Compute fundamental vectors associated with orbit

    This will compute the angular momentum, h, nodal, n, and eccentricity
    vector, e, for a Keplerian two-body orbit.

    Parameters
    ----------
    r : array_like and type
        Description of the variable

    Returns
    -------
    describe : type
        Explanation of return value named describe

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation of this parameter

    Raises
    ------
    BadException
        Because you shouldn't have done that.

    See Also
    --------
    other_func: Other function that this one might call

    Notes
    -----
    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] Shannon, Claude E. "Communication theory of secrecy systems."
    Bell Labs Technical Journal 28.4 (1949): 656-715

    Examples
    --------
    An example of how to use the function

    >>> a = [1, 2, 3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """

    # compute angular momentum vector
    mag_r = np.linalg.norm(r)
    mag_v = np.linalg.norm(v)
    rdotv = r.dot(v)

    h = np.cross(r, v)
    mag_h = np.linalg.norm(h)

    h_hat = h / mag_h

    # compute line of nodes vector
    n = np.zeros(3)
    n[0] = -h[1]
    n[1] = h[0]
    n[2] = 0.0

    if np.linalg.norm(n) < constants.small:
        mag_n = 0
        n_hat = np.zeros(3)
    else:
        mag_n = np.linalg.norm(n)
        n_hat = n / mag_n

    # compute eccentricity vector
    e = ((mag_v**2 - mu / mag_r) * r - rdotv * v) / mu
    if np.linalg.norm(e) < constants.small:
        e_hat = np.zeros(3)
    else:
        e_hat = e / np.linalg.norm(e)

    return h_hat, n_hat, e_hat


def rv2coe(r, v, mu):
    """Position and Velocity vectors to classical orbital elements

    [p,a,ecc,inc,raan,arg_p,nu,m,arglat,truelon,lonper] = rv2coe(r,v, mu)

    Inputs:
        - r - position vector in inertial frame (km)
        - v - velocity vector in inertial frame (km/sec)
        - mu - gravitational parameter of central body (km^3/sec^2)

    Outputs:
        - p - semi-major axis (km)
        - ecc - eccentricity
        - raan - right acsension of the ascending node (rad) 0 < raan <
        2*pi
        - inc - inclination (rad) 0 < inc < pi
        - arg_p - argument of periapsis (rad) 0 < arg_p < 2*pi
        - nu - true anomaly (rad) 0 < nu < 2*pi
        - m - mean anomaly in rad
        - arglat - argument of latitude(CI) rad 0 <arglat < 2*pi
        - truelon - true longitude (CE) rad 0 < truelon < 2*pi
        - lonper - longitude of periapsis rad 0 < lonper < 2*pi

    Dependencies:
        - numpy - everything is dependent on numpy
        - nu2anom - convert true anomaly to eccentric and mean anomaly

    Author:
        - Shankar Kulumani 30 Sept 2012
            - used old USAFA code and AAE532
        - Shankar Kulumani 5 September 2017
            - convert to Python for use in MAE3145

    References
        - AAE532 Notes
        - Vallado 3rd Edition
    """

    tol = 1e-9

    # calculate angular moemntum vector and magnitude
    mag_r = np.linalg.norm(r)
    mag_v = np.linalg.norm(v)

    rdotv = np.dot(r, v)

    h = np.cross(r, v)
    mag_h = np.linalg.norm(h)

    h_hat = h / mag_h
    # find n,e
    if mag_h > tol:
        n = np.zeros(3)
        n[0] = -h[1]
        n[1] = h[0]
        n[2] = 0.0

        with np.errstate(divide='raise'):
            try:
                mag_n = np.linalg.norm(n)
                n_hat = n / mag_n
            except FloatingPointError:
                mag_n = 0
                n_hat = np.zeros(3)

        # eccentricity vector
        e = ((mag_v**2 - mu / mag_r) * r - rdotv * v) / mu

        ecc = np.linalg.norm(e)
        mag_e = ecc

        sme = mag_v**2 / 2 - mu / mag_r

        if np.absolute(sme) > tol:
            a = -mu / (2 * sme)
        else:
            a = np.inf

        p = mag_h**2 / mu

        # inclination
        inc = np.arccos(h[2] / mag_h)

        # determine orbit type
        orbit_type = 'ei'
        if ecc < tol:
            # circular equatorial
            if inc < tol or np.absolute(inc - np.pi) < tol:
                orbit_type = 'ce'
            else:
                # circular inclined
                orbit_type = 'ci'

        else:
            #  elliptical, parabolic, hyperbolic equatorial -
            if inc < tol or np.absolute(inc - np.pi) < tol:
                orbit_type = 'ee'

        # right ascension of the ascending node
        if mag_n > tol:
            temp = n[0] / mag_n
            if np.absolute(temp) > 1.0:
                temp = np.sign(temp)
            raan = np.arccos(temp)
            if n[1] < 0.0:
                raan = 2 * np.pi - raan
        else:
            raan = 0

        # find argument of periapsis
        if orbit_type == 'ei':
            arg_p = np.arccos(np.dot(n, e) / (mag_n * mag_e))
            if e[2] < 0.0:
                arg_p = 2 * np.pi - arg_p
        else:
            arg_p = 0

        # ------------ find true anomaly at epoch - ------------
        if orbit_type[0] == 'e':
            nu = np.arccos(np.dot(e, r) / (mag_e * mag_r))
            if rdotv < 0.0:
                nu = 2 * np.pi - nu
        else:
            nu = 0

        # special orbit cases
        # ---- find argument of latitude - circular inclined - ----
        if orbit_type == 'ci':
            arglat = np.arccos(np.dot(n, r) / (mag_n * mag_r))
            if r[2] < 0.0:
                arglat = 2 * pi - arglat
            m = arglat
            nu = arglat
        else:
            arglat = 0

        # -- find longitude of perigee - elliptical equatorial - ---
        if ecc > tol and orbit_type == 'ee':
            temp = e[0] / ecc
            if np.absolute(temp) > 1.0:
                temp = np.sign(temp)
            lonper = np.arccos(temp)
            if e[1] < 0.0:
                lonper = 2 * pi - lonper
            if inc > pi / 2:
                lonper = 2 * pi - lonper

            arg_p = lonper
        else:
            lonper = 0

        # -------- find true longitude - circular equatorial - -----
        if mag_r > tol and orbit_type == 'ce':
            temp = r[0] / mag_r
            if np.absolute(temp) > 1.0:
                temp = np.sign(temp)
            truelon = np.arccos(temp)
            if r[1] < 0.0:
                truelon = 2 * np.pi - truelon
            if inc > np.pi / 2:
                truelon = 2 * np.pi - truelon
            m = truelon

            nu = truelon
        else:
            truelon = 0

        # find mean anomaly
        if orbit_type[0] == 'e':
            E, m = nu2anom(nu, ecc)
    else:
        p = 0
        a = 0
        ecc = 0
        inc = 0
        raan = 0
        arg_p = 0
        nu = 0
        m = 0
        arglat = 0
        truelon = 0
        lonper = 0

    return p, a, ecc, inc, raan, arg_p, nu, m, arglat, truelon, lonper


def kepler_eq_E(M_in, ecc_in):
    """Solve Kepler's Equation for all orbit types

    (E, nu, count) = kepler_eq_E(M, ecc)

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
       - count - number of iterations to converge

    Dependencies:
       - numpy - everything needs numpy
       - kinematics.attitude.normalize - normalize an angle

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
                    E_0 = M - np.sign(M) * ecc
                else:
                    E_0 = M / (ecc - 1.0)

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
            nu = np.arctan2(sinv, cosv)
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


def conic_orbit(p, ecc, inc, raan, arg_p, nu_i, nu_f, mu=constants.earth.mu):
    """Plot conic orbit

    Purpose:
        - Uses the polar conic equation to plot a conic orbit

    state_eci, state_pqw, state_lvlh, state_sat_eci, state_sat_pqw,
    state_sat_lvlh = conic_orbit(p,ecc,inc,raan,arg_p,nu_i,nu_f)

    Inputs:
        - p - semi-major axis (km)
        - ecc - eccentricity
        - raan - right acsension of the ascending node (rad) 0 < raan <
        2*pi
        - inc - inclination (rad) 0 < inc < pi
        - arg_p - argument of periapsis (rad) 0 < arg_p < 2*pi
        - nu_i - initial true anomaly (rad) 0 < nu < 2*pi
        - nu_f - final true anomaly (rad) 0 < nu < 2*pi
        - mu - gravitational paramter of central body (km^3/sec^2)

    Outputs:
        state_eci : (1000, 6) numpy array of satellite orbit in inertial frame
        state_pqw : (1000, 6) numpy array of satellite orbi in perifocal frame
        state_lvlh : (1000, 6) numpy array of satellite orbit in local
        veritical local horizontal frame
        state_sat_eci : (6,) numpy array of satellite state in inertial frame
        state_sat_pqw : (6,) numpy array of satellite state in perifocal frame
        state_sat_lvlh : (6,) numpy array of satellite state in local veritical and local horizontal fram

        the state is defined as 6 elements
        state[0:3] - position in reference frame in kilometer
        state[3:6] - velocity in reference frame in kilometer/second

    Dependencies:
        - ROT1,ROT2,ROT3 - principle axis rotation matrices

    Author:
        - Shankar Kulumani 1 Dec 2012
            - list revisions
        - Shankar Kulumani 5 Dec 2014
            - added outputs for orbit gui functions
        - Shankar Kulumani 5 Oct 2017
            - modify to include velocity and all three reference frames

    References
        - AAE532
        - MAE3145
    """

    tol = 1e-9
    step = 1000

    # v = true anomaly
    if nu_f > nu_i:
        v = np.linspace(nu_i, nu_f, step)
    else:
        v = np.linspace(nu_i, nu_f + 2 * np.pi, step)

    if ecc - 1 > tol:  # hyperbolic
        turn_angle = np.arccos(-1.0 / ecc)
        v = np.linspace(-turn_angle + 0.01, turn_angle - 0.01, step)

        if nu_i > np.pi:
            nu_i = nu_i - 2 * np.pi

        r = p / (1 + ecc * np.cos(v))
        rs = p / (1 + ecc * np.cos(nu_i))

    elif np.absolute(ecc - 1) < tol:  # parabolic
        v = np.linspace(-np.pi + 0.1, np.pi - 0.1, step)
        if nu_i > np.pi:
            nu_i = nu_i - 2 * np.pi

        r = p / 2 * (1 + np.tan(v / 2)**2)
        rs = p / 2 * (1 + np.tan(nu_i / 2)**2)
    else:
        # conic equation for elliptical orbit
        r = p / (1 + ecc * np.cos(v))
        rs = p / (1 + ecc * np.cos(nu_i))

    # position and velocity in local vertical and local horizontal frame
    pos_lvlh = np.stack((r, np.zeros_like(r), np.zeros_like(r)), axis=1)
    sat_lvlh = np.array([rs, 0, 0])

    vr = np.sqrt(mu / p) * ecc * np.sin(v)
    vt = np.sqrt(mu / p) * (1 + ecc * np.cos(v))

    vr_sat = np.sqrt(mu / p) * ecc * np.sin(nu_i)
    vt_sat = np.sqrt(mu / p) * (1 + ecc * np.cos(nu_i))

    v_lvlh = np.stack((vr, vt, np.zeros_like(vt)), axis=1)
    vs_lvlh = np.array([vr_sat, vt_sat, 0])

    # convert to perifocal reference frame
    pos_pqw = np.stack((r * np.cos(v),
                        r * np.sin(v),
                        np.zeros_like(r)), axis=1)
    sat_pqw = np.array([rs * np.cos(nu_i),
                        rs * np.sin(nu_i),
                        np.zeros_like(rs)])

    v_pqw = np.stack((np.cos(v) * vr - np.sin(v) * vt,
                      np.sin(v) * vr + np.cos(v) * vt,
                      np.zeros_like(vr)), axis=1)
    vs_pqw = np.array([np.cos(nu_i) * vr_sat - np.sin(nu_i) * vt_sat,
                       np.sin(nu_i) * vr_sat + np.cos(nu_i) * vt_sat,
                       0])
    # rotate orbit plane to inertial frame
    dcm_pqw2eci = attitude.rot3(-raan, 'r').dot(attitude.rot1(-inc, 'r')
                                                ).dot(attitude.rot3(-arg_p, 'r'))
    
    pos_eci = np.dot(dcm_pqw2eci, pos_pqw.T).T
    sat_eci = np.dot(dcm_pqw2eci, sat_pqw)

    v_eci = np.dot(dcm_pqw2eci, v_pqw.T).T
    vs_eci = np.dot(dcm_pqw2eci, vs_pqw)

    state_eci = np.concatenate((pos_eci, v_eci), axis=1)
    state_pqw = np.concatenate((pos_pqw, v_pqw), axis=1)
    state_lvlh = np.concatenate((pos_lvlh, v_lvlh), axis=1)

    state_sat_eci = np.concatenate((sat_eci, vs_eci))
    state_sat_pqw = np.concatenate((sat_pqw, vs_pqw))
    state_sat_lvlh = np.concatenate((sat_lvlh, vs_lvlh))

    return (state_eci, state_pqw, state_lvlh, state_sat_eci, state_sat_pqw,
            state_sat_lvlh)

# TODO: Add unit tests
def anom2nu(E, ecc):
    """Calculate true anomaly given eccentric anomaly for all orbits

    (nu) = anom2nu(E,ecc) 

    Inputs:
        - ecc - eccentricity of orbit 0 < ecc < inf
        - E - (elliptical/parabolic/hyperbolic) eccentric anomaly in rad
            0 < E < 2*pi

    Outputs:
        - nu - true anomaly in rad -2*pi < nu < 2*pi

    Dependencies:
        - numpy - we are lost without numpy
        - kinematics.attitude.normalize - normalize angles
    
    Notes
    -----
    This function is valid for all orbit types. 

    Author:
        - Shankar Kulumani 23 Nov 2017
            - Add function in python

    References
        - AAE532 notes
        - Vallado 3rd Ed
    """
    small = 1e-9

    if ecc <= small:  # circular
        nu = E
    elif small < ecc and ecc <= 1 - small:  # elliptical
        sinv = np.sin(E) * np.sqrt(1- ecc**2) / (1 - ecc * np.cos(E))
        cosv = (np.cos(E) - ecc) / (1 - ecc * np.cos(E))
        nu = np.arctan2(sinv, cosv)

        nu = attitude.normalize(nu, 0, 2 * np.pi)
    elif np.absolute(ecc - 1) <= small:  # parabolic
        B = E
        sinv = p * B / r
        cosv = (p - r) / r

        nu = np.arctan2(B)
        
        nu = attitude.normalize(nu, 0, 2 * np.pi)
    elif ecc > 1 + small:  # hyperbolic
        H = E
        sinv = -np.sinh(H) * np.sqrt(ecc**2 - 1) / (1 - ecc * np.cosh(H))
        cosv = (np.cosh(H) - ecc) / (1 - ecc * np.cosh(H))

        nu = np.arctan2(sinv, cosv)
    else:
        print("Eccentricity is out of bounds 0 < ecc < inf")

    return nu[0]


# TODO: Add unit tests
# TODO: Use arctan2 formulation of conversion
def nu2anom(nu, ecc):
    """Calculates the eccentric and mean anomaly given eccentricity and true
    anomaly

    ( E, M ) = ecc_anomaly(nu,ecc)

    Inputs:
        - nu - true anomaly in rad -2*pi < nu < 2*pi
        - ecc - eccentricity of orbit 0 < ecc < inf

    Outputs:
        - E - (elliptical/parabolic/hyperbolic) eccentric anomaly in rad
            0 < E < 2*pi
        - M - mean anomaly in rad 0 < M < 2*pi

    Dependencies:
        - numpy - we are lost without numpy
        - kinematics.attitude.normalize - normalize angles
    
    Notes
    -----
    This function is valid for all orbit types. 

    Author:
        - Shankar Kulumani 20 Nov 2017
            - only now realized I already implemented other orbit types
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
        # TODO: Need to check if I need to do quadrant checks for parabolic or hyperbolic cases
        # E = revcheck(E);
        # M = revcheck(M);
    elif ecc > 1 + small:  # hyperbolic
        sine = (np.sqrt(ecc**2 - 1) * np.sin(nu)) / \
            (1.0 + ecc * np.cos(nu))
        cose = (ecc + np.cos(nu)) / (1 + ecc * np.cos(nu))
        H = np.arctan2(sine, cose)

        E = H
        M = ecc * np.sinh(H) - H

        # E = revcheck(E);
        # M = revcheck(M);
    else:
        print("Eccentricity is out of bounds 0 < ecc < inf")

    return (E, M)


def tof_delta_t(p, ecc, mu, nu_0, delta_t):
    """Use time of fligt to compute future true anomaly for eccentric orbits

       Calculate change in orbital position (true anomaly) given a delta_t
       change in time

    Inputs: 
        - p - semi-latus rectum in km
        - ecc - eccentricity of orbit 0 < ecc < 1
        - mu - gravitational parameter
        - nu_0 - initial true anomaly in rad 0 < nu_0 < 2*pi
        - delta_t - change in time in seconds
        
    Outputs: 
        - nu_f - true anomaly after delta_t in rad 0 < nu_f < 2*pi

    Dependencies: 
        - ecc_anomaly.m - calculates eccentric and mean anomaly from true
        anomaly
        - kepler_eq_E.m - solves for eccentric anomaly and true anomaly
        given a mean anomaly

    Author: 
        - Shankar Kulumani 15 Sept 2012
            - written using USAFA code and AAE532 PS4 homework
        - Shankar Kulumani 19 Sept 2012
            - added E and M outputs
        - Shankar Kulumani 1 Oct 2012
            - added semi-latus rectum input
        - Shankar Kulumani 9 Oct 2017
            - now in Python

    References
        - Vallado 3rd Edition
        - AAE532 Notes
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

def tof_nu(p, ecc, nu_1, nu_2, mu=constants.earth.mu):
    """Calculate TOF between two known true anomaly positions

    Inputs: 
        - p - semi-major axis (or semi-parameter for parabolic) in km
        - ecc - eccentricity 0 < ecc < 1
        - mu - gravitational parameter of central body in km^3/sec^2
        - nu_0 - initial true anomaly in rad 0 < nu_0 < 2*pi
        - nu_f - final true anomaly in rad 0 < nu_f < 2*pi

    Outputs: 
        - tof - time of flight in seconds

    Dependencies: 
        - ecc_anomaly - calculates eccentric anomaly from true anomlay

    Author: 
        - Shankar Kulumani 16 Sept 2012
            - uses code from AAE532 PS4 in function form
        - Shankar Kulumani 29 Sept 2012
            - added eccentricity logic for different orbit types
        - Shankar Kulumani 1 Oct 2012
            - added semi-latus rectum instead of semi-major axis
        - Shankar Kulumani  11 Oct 2017
            - convert to Python

    Note:
        Currently only supports elliptical orbits. Need to modify

    References
        - AAE532 Notes
        - Vallado 3rd Edition
        - Bate,Mueller,White
    """

    tol = constants.small

    # calculate eccentric anomaly from true anomaly
    #TODO: need to modify to handle other orbit types
    E_0, M_0 = nu2anom(nu_1,ecc)

    E_f, M_f = nu2anom(nu_2,ecc)


    # check eccentricity
    if np.absolute(ecc-1) < tol: # parabolic
        
        n = 2*np.sqrt(mu/p**3)
    elif ecc > 1+tol:
        a = p/(ecc**2-1)
        # calculate mean motion
        n = np.sqrt(mu/a**3)
    else:
        a = p/(1-ecc**2)
        # calculate mean motion
        n = np.sqrt(mu/a**3)

    # calculate time of flight

    tof = (M_f-M_0)/n
    
    return tof

def fpa_solve(nu, ecc):
    """Calculate flight path angle

    Inputs:
        - nu - true anomaly of orbit -2*pi < nu < 2*pi
        - ecc - eccentricity of orbit 0 < ecc < inf

    Outputs:
        - fpa - flight path angle in rad -pi < fpa < pi

    Dependencies:
        - none

    Author:
        - Shankar Kulumani 15 Sept 2012
            - created for AAE532 PS4
        - Shankar Kulumani 29 Sept 2012
            - modified for diferennt orbit types

    References
        - AAE532 Notes
        - Vallado 3rd Ed pg 113
    """
    tol = 1e-9

    if ecc - 1 > tol:  # hyperbolic
        # convert nu to eccentric anomaly
        H, _ = nu2anom(nu, ecc)
        sin_fpa = ecc * np.sinh(H) / np.sqrt(ecc**2 * np.cosh(H)**2 - 1)
        cos_fpa = np.sqrt((ecc**2 - 1) / (ecc**2 * np.cosh(H)**2 - 1))
        fpa = np.arctan2(sin_fpa, cos_fpa)  # rad
    else:
        if np.absolute(ecc - 1) < tol:  # parabolic
            fpa = nu / 2
        else:
            if ecc > tol:  # elliptical
                sin_fpa = ecc * np.sin(nu) / np.sqrt(1 +
                                                     2 * ecc * np.cos(nu) + ecc**2)
                cos_fpa = (1 + ecc * np.cos(nu)) / \
                    np.sqrt(1 + 2 * ecc * np.cos(nu) + ecc**2)

                # atan2 allows for quadrant check
                fpa = np.arctan2(sin_fpa, cos_fpa)  # rad
            else:  # circular
                fpa = 0

    return fpa


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
    a = p / (1 - ecc**2)  # km
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
    
    # TODO: Add a function that compute pos/vel in LVLH frame alone
    # convert position and velocity to lvlh frame
    r_lvlh = np.array([r, 0, 0])  # [r_hat theta_hat h_hat] km
    # [r_hat theta_hat h_hat] km/sec
    v_lvlh = v * np.array([np.sin(fpa), np.cos(fpa), 0])

    v_circ = np.sqrt(mu / r)

    v_esc = np.sqrt(2) * v_circ

    E, M = nu2anom(nu, ecc)  # rad
    E = float(E)
    M = float(M)

    n = np.sqrt(mu / a**3)  # mean motion in 1/sec

    return (a, h, period, sme, fpa, r_per, r_apo, r_ijk, v_ijk, r_pqw, v_pqw,
            r_lvlh, v_lvlh, r, v, v_circ, v_esc, E, M, n)


def par_orbit_el(p, ecc, inc, raan, arg_p, nu, mu):

    # calculate semi-latus rectum (km)
    a = np.inf
    # velocity at infinite radius (km/sec)
    v_inf = 0

    # mechanical energy (km^2/sec^2)
    sme = mu / 2 / a

    # angular momentum scalar
    h = np.sqrt(p * mu)  # km^2/sec

    fpa = fpa_solve(nu, ecc)  # radians

    # radius of periapsis and apoapsis
    r_per = p / 2  # km

    r_ijk, v_ijk, r_pqw, v_pqw = coe2rv(p, ecc, inc, raan, arg_p, nu, mu)

    # position and velocity within orbit
    r = norm(r_pqw)  # km
    v = norm(v_pqw)  # km/sec

    # convert position and velocity to lvlh frame
    r_lvlh = np.array([r, 0, 0])  # [r_hat theta_hat h_hat] km
    # [r_hat theta_hat h_hat] km/sec
    v_lvlh = v * np.array([np.sin(fpa), np.cos(fpa), 0])

    v_circ = np.sqrt(mu / r)

    v_esc = np.sqrt(2) * v_circ

    B, M_B = nu2anom(nu, ecc)  # rad

    n = 2 * np.sqrt(mu / p**3)  # mean motion in 1/sec

    return (a, v_inf, sme, h, fpa, r_per, r_ijk, v_ijk,
            r_pqw, v_pqw, r_lvlh, v_lvlh, r, v, v_circ, v_esc, B, M_B, n)


def hyp_orbit_el(p, ecc, inc, raan, arg_p, nu, mu):
    """Hyperbolic Orbit Characteristics/Elements

    Purpose: 
        - calculates orbital parameters for a hyperbolic orbit

    [a, v_inf, b, sme, flyby, nu_inf, h, fpa, r_per, r_ijk, v_ijk, 
    r_pqw, v_pqw, r_lvlh, v_lvlh, r, v, v_circ, v_esc, H, M_H, n] = hyp_orbit_el(
    p, ecc, inc, raan, arg_p, nu, mu)

    Inputs: 
        - 

    Outputs: 
        - List/describe outputs of function

    Dependencies: 
        - coe2rv.m - convert COE to position and velocity vector

    Author: 
        - Shankar Kulumani 29 Sept 2012
            - modified outputs
        - Shankar Kulumani 5 September 2017
            - convert to Python for MAE3145

    References
        - AAE532 and any astrodynamics book 
    """
    # calculate semi-latus rectum (km)
    a = p / (ecc**2 - 1)
    # velocity at infinite radius (km/sec)
    v_inf = np.sqrt(mu / a)
    # aiming radius (semi-minor axis b) (km)
    b = a * np.sqrt(ecc**2 - 1)
    # mechanical energy (km^2/sec^2)
    sme = mu / 2 / a
    # flyby angle (rad)
    flyby = np.arcsin(1 / ecc) * 2
    # true anomaly at infinite radius (rad)
    nu_inf = np.pi / 2 + flyby / 2

    # angular momentum scalar
    h = np.sqrt(p * mu)  # km^2/sec

    fpa = fpa_solve(nu, ecc)  # radians

    # radius of periapsis and apoapsis
    r_per = a * (ecc - 1)  # km

    r_ijk, v_ijk, r_pqw, v_pqw = coe2rv(p, ecc, inc, raan, arg_p, nu, mu)

    # position and velocity within orbit
    r = np.linalg.norm(r_pqw)  # km
    v = np.linalg.norm(v_pqw)  # km/sec

    # convert position and velocity to lvlh frame
    r_lvlh = np.array([r, 0, 0])  # [r_hat theta_hat h_hat] km
    # [r_hat theta_hat h_hat] km/sec
    v_lvlh = v * np.array([np.sin(fpa), np.cos(fpa), 0])

    v_circ = np.sqrt(mu / r)

    v_esc = np.sqrt(2) * v_circ

    H, M_H = nu2anom(nu, ecc)  # rad

    n = np.sqrt(mu / a**3)  # mean motion in 1/sec

    return (a, v_inf, b, sme, flyby, nu_inf, h, fpa, r_per, r_ijk, v_ijk,
            r_pqw, v_pqw, r_lvlh, v_lvlh, r, v, v_circ, v_esc, H, M_H, n)

# TODO: Output eccentricity, ang mom, and nodal vectors
def orbit_el(p, ecc, inc, raan, arg_p, nu, mu, print_flag=False):
    """Orbit Characteristics/Elements

    Purpose:
        - Calculates  orbital parameters using conic equations

        orbit_el(p,ecc,inc,raan,arg_p,nu,mu,print_flag)

    Inputs:
        - a - semi-major axis in km
        - ecc - eccentricity of orbit
        - mu - gravitational parameter in km^3/sec^2
        - nu - true anomaly in rad 0 < nu < 2*pi
        - print_flag - 'true' or 'false' to print outputs to screen

    Outputs:
        - none - prints data to screen

    Dependencies:
        - elp_orbit_el.m - elliptical orbit elements
        - hyp_orbit_el.m - hypberbolic orbit elements
        - par_orbit_el.m - parabolic orbit elements

    Author:
        - Shankar Kulumani 16 Sept 2012
            - used code from AAE532 PS4
        - Shankar Kulumani 19 Sept 2012
            - added escape speed
            - added print capability
            - added constants
            - added mean motion
            - added lvlh and pqw vectors
        - Shankar Kulumani 27 Sept 2012
            - modifying to add hyperbolic orbit capability
            - moved elliptical stuff to elp_orbit_el.m
            - removed outputs
        - Shankar Kulumani 29 Sept 2012
            - modified fprintf commands to make it more like STK
        - Shankar Kulumani 5 September 2017
            - modify for Python for MAE3145

    References
        - AAE532 Notes

    """
    # load constants

    r_earth = constants.earth.radius
    km2au = constants.km2au
    rad2deg = constants.rad2deg
    tol = 1e-9

    if ecc < 1:  # elliptical
        (a, h, period, sme, fpa, r_per, r_apo, r_ijk, v_ijk,
         r_pqw, v_pqw, r_lvlh, v_lvlh, r, v, v_circ, v_esc,
         E, M, n) = elp_orbit_el(p, ecc, inc, raan, arg_p, nu, mu)
        # build string for output
        string = '\n'

        string += 'Satellite State \n'
        string += 'Position and Velocity in LVLH frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'r_hat:', r_lvlh[0], 'rd_hat:', v_lvlh[0], 't_hat:', r_lvlh[1], 'td_hat:', v_lvlh[1], 'h_hat:', r_lvlh[2], 'hd_hat:', v_lvlh[2])

        string += '\n'
        string += 'Position and Velocity in EPH/PQW frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'e_hat:', r_pqw[0], 'ed_hat:', v_pqw[0], 'p_hat:', r_pqw[1], 'pd_hat:', v_pqw[1], 'h_hat:', r_pqw[2], 'hd_hat:', v_pqw[2])

        string += '\n'
        string += 'Position and Velocity in IJK frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'i_hat:', r_ijk[0], 'id_hat:', v_ijk[0], 'j_hat:', r_ijk[1], 'jd_hat:', v_ijk[1], 'k_hat:', r_ijk[2], 'kd_hat:', v_ijk[2])

        string += '\n'
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'RAD_MAG', r, r * km2au)
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_MAG', v)

        string += '\n'
        string += 'Orbital Elements \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} deg\n{:10s} {:20.15g}    {:10s} {:20.15g} deg\n{:10s} {:20.15g} deg {:9s} {:20.15g} deg\n'.format(
            'sma:', a, 'raan:', raan * rad2deg, 'ecc:', ecc, 'arg_p:', arg_p * rad2deg, 'inc:', inc * rad2deg, 'nu:', nu * rad2deg)

        string += '\n'
        string += 'Elliptic Orbital Parameters\n'
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'P', p, p * km2au)
        string += '{:10s}: {:20.15g} km^2/sec \n'.format('ANG MOM', h)
        string += '{:10s}: {:20.15g} sec = {:20.15g} hr\n'.format(
            'PERIOD', period, period / 3600)
        string += '{:10s}: {:20.15g} km^2/sec^2 \n'.format('ENGERGY', sme)
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'RAD_PER', r_per, r_per * km2au)
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'RAD_APO', r_apo, r_apo * km2au)

        string += '\n'
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_CIRC', v_circ)
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_ESC', v_esc)
        string += '{:10s}: {:20.15g} deg \n'.format('TRUE_ANOM', nu * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('FPA', fpa * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('ECC_ANOM', E * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('MEAN_ANOM', M * rad2deg)
        string += '{:10s}: {:20.15g} deg/sec \n'.format(
            'MEAN_MOT', n * rad2deg)

        string += '\n'
        string += '{:10s}: {:20.15g} sec = {:20.15g} hr \n'.format(
            'T_PAST_PER', 1 / n * M, 1 / n * M / 3600)
    elif (ecc - 1) > tol:  # hyperbolic
        (a, v_inf, b, sme, flyby, nu_inf, h, fpa, r_per, r_ijk, v_ijk, r_pqw,
         v_pqw, r_lvlh, v_lvlh, r, v, v_circ, v_esc, H, M_H, n) = hyp_orbit_el(p, ecc, inc, raan, arg_p, nu, mu)
        # build string for output
        string = '\n'

        string += 'Satellite State \n'
        string += 'Position and Velocity in LVLH frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'r_hat:', r_lvlh[0], 'rd_hat:', v_lvlh[0], 't_hat:', r_lvlh[1], 'td_hat:', v_lvlh[1], 'h_hat:', r_lvlh[2], 'hd_hat:', v_lvlh[2])

        string += '\n'
        string += 'Position and Velocity in EPH/PQW frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'e_hat:', r_pqw[0], 'ed_hat:', v_pqw[0], 'p_hat:', r_pqw[1], 'pd_hat:', v_pqw[1], 'h_hat:', r_pqw[2], 'hd_hat:', v_pqw[2])

        string += '\n'
        string += 'Position and Velocity in IJK frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'i_hat:', r_ijk[0], 'id_hat:', v_ijk[0], 'j_hat:', r_ijk[1], 'jd_hat:', v_ijk[1], 'k_hat:', r_ijk[2], 'kd_hat:', v_ijk[2])

        string += '\n'
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'RAD_MAG', r, r * km2au)
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_MAG', v)

        string += '\n'
        string += 'Orbital Elements \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} deg\n{:10s} {:20.15g}    {:10s} {:20.15g} deg\n{:10s} {:20.15g} deg {:9s} {:20.15g} deg\n'.format(
            'sma:', a, 'raan:', raan * rad2deg, 'ecc:', ecc, 'arg_p:', arg_p * rad2deg, 'inc:', inc * rad2deg, 'nu:', nu * rad2deg)

        string += '\n'
        string += 'Hyperbolic Orbital Parameters\n'
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'P', p, p * km2au)
        string += '{:10s}: {:20.15g} km^2/sec \n'.format('ANG MOM', h)
        string += '{:10s}: {:20.15g} km^2/sec^2 \n'.format('ENGERGY', sme)
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'RAD_PER', r_per, r_per * km2au)

        string += '\n'
        string += '{:10s}: {:20.15g} km/sec \n'.format('V_INF', v_inf)
        string += '{:10s}: {:20.15g} km \n'.format('RAD_AIM', b)
        string += '{:10s}: {:20.15g} deg \n'.format('FLYBY', flyby * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('NU_INF', nu_inf * rad2deg)

        string += '\n'
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_CIRC', v_circ)
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_ESC', v_esc)
        string += '{:10s}: {:20.15g} deg \n'.format('TRUE_ANOM', nu * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('FPA', fpa * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('HYP_ANOM', H * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('MEAN_ANOM', M_H * rad2deg)
        string += '{:10s}: {:20.15g} deg/sec \n'.format(
            'MEAN_MOT', n * rad2deg)

        string += '\n'
        string += '{:10s}: {:20.15g} sec = {:20.15g} hr \n'.format(
            'T_PAST_PER', 1 / n * M_H, 1 / n * M_H / 3600)
    elif np.absolute(ecc - 1) < tol:  # parabolic
        (a, v_inf, sme, h, fpa, r_per, r_ijk, v_ijk, r_pqw, v_pqw, r_lvlh,
         v_lvlh, r, v, v_circ, v_esc, B, M_B, n) = par_orbit_el(p, ecc, inc,
                                                                raan, arg_p,
                                                                nu, mu)
        # build string for output
        string = '\n'

        string += 'Satellite State \n'
        string += 'Position and Velocity in LVLH frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'r_hat:', r_lvlh[0], 'rd_hat:', v_lvlh[0], 't_hat:', r_lvlh[1], 'td_hat:', v_lvlh[1], 'h_hat:', r_lvlh[2], 'hd_hat:', v_lvlh[2])

        string += '\n'
        string += 'Position and Velocity in EPH/PQW frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'e_hat:', r_pqw[0], 'ed_hat:', v_pqw[0], 'p_hat:', r_pqw[1], 'pd_hat:', v_pqw[1], 'h_hat:', r_pqw[2], 'hd_hat:', v_pqw[2])

        string += '\n'
        string += 'Position and Velocity in IJK frame \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n{:10s} {:20.15g} km {:10s} {:20.15g} km/sec\n'.format(
            'i_hat:', r_ijk[0], 'id_hat:', v_ijk[0], 'j_hat:', r_ijk[1], 'jd_hat:', v_ijk[1], 'k_hat:', r_ijk[2], 'kd_hat:', v_ijk[2])

        string += '\n'
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'RAD_MAG', r, r * km2au)
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_MAG', v)

        string += '\n'
        string += 'Orbital Elements \n'
        string += '{:10s} {:20.15g} km {:10s} {:20.15g} deg\n{:10s} {:20.15g}    {:10s} {:20.15g} deg\n{:10s} {:20.15g} deg {:9s} {:20.15g} deg\n'.format(
            'sma:', a, 'raan:', raan * rad2deg, 'ecc:', ecc, 'arg_p:', arg_p * rad2deg, 'inc:', inc * rad2deg, 'nu:', nu * rad2deg)

        string += '\n'
        string += 'Parabolic Orbital Parameters\n'
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'P', p, p * km2au)
        string += '{:10s}: {:20.15g} km^2/sec \n'.format('ANG MOM', h)
        string += '{:10s}: {:20.15g} km^2/sec^2 \n'.format('ENGERGY', sme)
        string += '{:10s}: {:20.15g} km = {:20.15g} AU \n'.format(
            'RAD_PER', r_per, r_per * km2au)

        string += '\n'
        string += '{:10s}: {:20.15g} km/sec \n'.format('V_INF', v_inf)

        string += '\n'
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_CIRC', v_circ)
        string += '{:10s}: {:20.15g} km/sec \n'.format('VEL_ESC', v_esc)
        string += '{:10s}: {:20.15g} deg \n'.format('TRUE_ANOM', nu * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('FPA', fpa * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('HYP_ANOM', B * rad2deg)
        string += '{:10s}: {:20.15g} deg \n'.format('MEAN_ANOM', n * rad2deg)
        string += '{:10s}: {:20.15g} deg/sec \n'.format(
            'MEAN_MOT', n * rad2deg)

        string += '\n'
        string += '{:10s}: {:20.15g} sec = {:20.15g} hr \n'.format(
            'T_PAST_PER', 1 / n * M_B, 1 / n * M_B / 3600)

    if print_flag:
        print(string)

    return string

def semilatus_rectum(a, ecc):
    r"""Compute the semilatus rectum

    Given the semimajor axis and eccentricty, this will compute the semilatus rectum.

    Parameters
    ----------
    a : float
        Semimajor axis (distance unit)
    ecc : float
        Eccentricity of orbit (unitless)

    Returns
    -------
    p : float
        Semilatus rectum in the same units as a

    Notes
    -----

    .. math:: p = a ( 1 - e^2)

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu

    References
    ----------

    .. [1] Fundamentals of Astrodynamics

    """
    p = a * (1 - ecc**2)
    
    if hasattr(ecc, "__iter__"):
        p[ np.absolute(ecc - 1) < 1e-6] = 0
    else:
        if np.absolute(ecc-1) < 1e-6:
            p = 0

    return p

def hyp_per2sma(rp, ecc):
    r"""Convert periapsis to semimajor axis for hyperbolic orbits

    Determine semi-major axis and semi-latus rectum for hyperbolic orbits

    Parameters
    ----------
    rp : float
        Periapsis distance in kilometers or other distance unit
    ecc : float
        Eccentricty of orbit - should be greater than 1

    Returns
    -------
    a : float
        Semimajor axis in kilometers
    p : float
        Semilatus rectum in same units as input distance

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] Shannon, Claude E. "Communication theory of secrecy systems."
    Bell Labs Technical Journal 28.4 (1949): 656-715

    """
    
    if ecc <= 1:
        print("Eccentricty should be greater than 1")
        return 1

    a = - rp / (ecc - 1)
    p = semilatus_rectum(a, ecc)
    return a, p

# TODO: add documentation
def nu_solve(p, e, r):
    """Solve conic equation for true anomaly
    
    nu, nu_neg = nu_solv(p, ecc, r)

    Parameters
    ----------
    p : float
        Semi parameter
    e : float
        eccentricity
    r : float
        radius of orbit

    Returns
    -------
    nu : float
        True anomaly in radians
    nu_neg : float
        Negative true anomaly in radians

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu
    """

    nu = np.arccos( p / r / e - 1 / e)
    return nu, -nu

# TODO NEed unit test
def fg_propogate(r_old, v_old, nu_old, nu_new, p, ecc, mu):
    """F and G relationship propogator

    Purpose: 
        - Find new position and velocity vectors in inertial frame using f
        and g relationships

            [r_new v_new f g f_dot g_dot delta_nu] = fandg_nu(r_old,v_old,nu_old,nu_new,p,ecc,mu)

    Inputs: 
        - r_old - position vector in inertial frame in km
        - v_old - velocity vecotr in inertial frame in km/sec
        - nu_old - true anomaly at initial condition in rad
        - nu_new - true anomaly at final position in rad
        - p - semi-latus rectum of orbit in km
        - ecc - eccentricity of orbit
        - mu - gravitational parameter of central body in km^3/sec^2

    Outputs: 
        - r_new - new position vector (3,)
        - v_new - new velocity vector (3,)
        - f - f function value
        - g - g function value
        - f_dot - fdot value
        - g_dot - gdot value
        - delta_nu - change in true anomaly

    Dependencies: 
        - None

    Author: 
        - Shankar Kulumani 13 Oct 2012
            - list revisions
        - Shankar Kulumani 13 December 2017
            - moving to python

    References
        - AAE532 LSN 14 notes
        - AAE532_PS7.pdf
    """
    delta_nu = nu_new-nu_old 

    r = p/(1+ecc*np.cos(nu_new))
    r0 = np.norm(r_old)

    f = 1-r/p*(1-cos(delta_nu))
    g = (r*r0/np.sqrt(mu*p))*np.sin(delta_nu)

    f_dot = (np.dot(r_old,v_old)/(p*r0)*(1-np.cos(delta_nu)))-(1/r0*np.sqrt(mu/p)*np.sin(delta_nu))
    g_dot = 1-r0/p*(1-np.cos(delta_nu))

    r_new = f*r_old +g*v_old
    v_new = f_dot*r_old + g_dot*v_old
    
    return r_new, v_new, f, g, f_dot, g_dot, delta_nu

# TODO Add unit tests
def fg_velocity(r1, r2, delta_nu, p, mu):
    """F and G function using delta true anomaly

    Purpose: 
        - Solves for new velocity vectors using f and g functions adn a
        change in true anomaly

    [v1 v2 f g f_dot g_dot] = fg_nu(r1,r2,dnu,p,mu)

    Inputs: 
        - r1 - initial position vector (1x3 or 3x1) in km
        - r2 - final position vector ( same size as r1) in km
        - dnu - delta true anomaly between r1 and r2
        - p - semi-parameter of orbit in km
        - mu - gravitational parameter in km^3/sec^2

    Outputs: 
        - v1 - initial velocity vector in km/sec
        - v2 - final velocity vector in km/sec
        - f - f function 
        - g - g function
        - f_dot - f dot function
        - g_dot - g dot function

    Dependencies: 
        - none

    Author: 
        - Shankar Kulumani 5 Nov 2012
            - list revisions

    References
        - AAE532 LSN 18  
    """
    r = np.linalg.norm(r2)
    r0 = np.linalg.norm(r1)

    f = 1-(r/p)*(1-np.cos(delta_nu))
    g = (r*r0/np.sqrt(mu*p))*np.sin(delta_nu)

    v1 = (r2-f*r1)/g

    f_dot = (np.dot(r1,v1)/(p*r0)*(1-np.cos(delta_nu)))-(1/r0*np.sqrt(mu/p)*np.sin(delta_nu))
    g_dot = 1-r0/p*(1-np.cos(delta_nu))

    v2 = f_dot*r1+g_dot*v1

    return v1, v2, f, g, f_dot, g_dot

# TODO Add unit tests
def period2sma(period, mu):
    """Convert period to semi major axis
    """
    a = ((period / 2 / np.pi)**2 * mu )**(1/3)
    return a

# TODO Documetnation and unit tests
def n2a(n, mu):
    """Convert mean motion to semi major axis
    """
    a = (mu / n**2)**(1/3)
    return a

# TODO Documenation and unit tests
def a2n(a, mu):
    """Convert semi major axis to mean motion
    """
    n = np.sqrt(mu/a**3)
    return n
