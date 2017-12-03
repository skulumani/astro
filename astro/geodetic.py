"""Geodetic transformations. This module holds many functions to peform common
transformations used in astrodynamics.

Copyright (C) {2017}  {Shankar Kulumani}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np
from kinematics import attitude
from astro import time, constants, transform
import pdb
# TODO: Make these funcitons vectorized to handle vector inputs


def lla2ecef(lat, lon, alt, r=6378.137, ee=8.1819190842622e-2):
    """Convert latitude, longitude, and altitude to the Earth centered
    Earth fixed frame (ECEF)

    Will convert between geodetic latitude, longitude, and altitude to the
    associated planet fixed reference frame.  This assumes an oblate spheroid
    model for the planetary body. It is applicable to any spherical body given
    the semi-major axis and the body eccentricity constant.

    Parameters:
    -----------
    lat : float
        Geodetic latitude (radians)
    lon : float
        Geodetic longitude (radians)
    alt : float
        height about the mean surface ellipsoid (WGS84 for Earth) (kilometers)

    Returns
    -------
    ecef : float ndarray (3,)
        Body fixed position vector in (kilometers)

    Notes
    -----
    We're assuming an oblate spheroid model for the Earth. To transform this
    vector to the inertial ECI frame all you need to do is rotate it by the
    Greenwich Mean Sideral time (GST) or use a fancy ECEF-ECI rotation which
    accounts for the Earth's precession/nutation.

    References
    ----------
    .. [1] BATE, Roger R, MUELLER, Donald D WHITE, Jerry E. Fundamentals of
    Astrodynamics. Courier Dover Publications, 1971.

    .. [2] Nice website to verify computations:
    http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm

    Examples
    --------
    Some examples demonstrating the usage of this function

    >>> import numpy

    Convert the lattitude, longitude, and altitude of Washington, DC to it's
    equivalent ECEF representation.

    >>> lat, lon, alt = (38.8895 * np.pi / 180, -77.035 * np.pi / 180, 0.054)

    We need to make sure the inputs are in the correct units.

    >>> lla2ecef(lat, lon, alt)
    [1115.308, -4844.546, 3982.965]

    """
    # Normal distance from teh surface to the Z axis along the ellipsoid normal
    N = r / np.sqrt(1 - ee**2 * np.sin(lat)**2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - ee**2) * N + alt) * np.sin(lat)

    return np.array([x, y, z])



# TODO: Change documentation to ECI (inertial frame) and verify


def site2eci(lat, alt, lst, r=6378.137, ee=8.1819190842622e-2):
    """Calculate the site vector in the ECI coordinate system.

    Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741

    Inputs:
        sitlat - site latitude (radians)
        lst - local sidereal time (radians)
        sitalt - site altitude (meters)

    Outputs:
        R_site - site vector in ECI frame - inertial frame

    Globals:
        RE, EEsqrd

    Constants: None

    Coupling: None

    References:
        Astro 321 COMFIX
        Vallado Alg 47
    """
    N = r / np.sqrt(1 - ee**2 * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat)
    z = ((1 - ee**2) * N + alt) * np.sin(lat)
    eci = np.array([x * np.cos(lst), x * np.sin(lst), z])

    return eci

# TODO:Add documentation


def ecef2lla(ecef, r=6378.137, ee=8.1819190842622e-2):
    """Converts a ECEF vector to the equivalent Lat, longitude and Altitude
    above the reference ellipsoid

    """
    twopi = 2 * np.pi
    tol = 1e-6

    norm_vec = np.linalg.norm(ecef)

    temp = np.sqrt(ecef[0]**2 + ecef[1]**2)
    if temp < tol:
        rtasc = np.sign(ecef[2]) * np.pi * 0.5
    else:
        rtasc = np.arctan2(ecef[1], ecef[0])

    lon = rtasc
    lon = attitude.normalize(lon, 0, 2 * np.pi)

    decl = np.arcsin(ecef[2] / norm_vec)
    latgd = decl

    # iterate to find geodetic latitude
    i = 1
    olddelta = latgd + 10.0
    while np.absolute(olddelta - latgd) >= tol and i < 10:
        oldelta = latgd
        sintemp = np.sin(latgd)
        c = r / np.sqrt(1.0 - ee**2 * sintemp**2)
        latgd = np.arctan2(ecef[2] + c * ee**2 * sintemp, temp)
        i = i + 1

    # calculate the height
    if (np.pi / 2 - np.absolute(latgd)) > np.pi / 180:
        hellp = temp / np.cos(latgd) - c
    else:
        s = c * (1 - ee**2)
        hellp = ecef[2] / np.sin(latgd) - s

    latgc = gd2gc(latgd, ee**2)
    return latgc, latgd, lon, hellp


def gd2gc(latgd, eesqrd=0.081819221456**2):
    """Only valid for locations on the Earth's surface

    Vallado Example 3-1
    """
    latgc = np.arctan((1 - eesqrd) * np.tan(latgd))
    return latgc


def gc2gd(latgc, eesqrd=0.081819221456**2):
    """Only valid for locations on the Earth's surface

    Vallado Example 3-1
    """
    latgd = np.arctan(np.tan(latgc) / (1 - eesqrd))
    return latgd

# TODO: Incorporate all the other observation angles (RADEC, etc. Vallado Ch 4)


def rv2rhoazel(r_sat_eci, v_sat_eci, lat, lon, alt, jd):
    """
    This function calculates the topcentric range,azimuth and elevation from
    the site vector and satellite position vector.

    Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741

    Inputs:
        sat_eci - satellite ECI position vector (km)
        site_eci - site ECI position vector (km)
        site_lat - site geodetic latitude (radians)
        site_lst - site local sidereal time (radians)

    Outputs:
        rho - range (km)
        az - asimuth (radians)
        el - elevation (radians)

    Globals: None

    Constants: None

    Coupling:

    References:
        Astro 321 Predict LSN 22
        Vallado Algorithm 27
    """
    small = constants.small
    halfpi = constants.halfpi

    # get site in ECEF
    r_site_ecef = lla2ecef(lat, lon, alt)

    # convert sat and site to ecef
    dcm_eci2ecef = transform.dcm_eci2ecef(jd)
    omega_earth = np.array([0, 0, constants.earth.omega])

    r_sat_ecef = dcm_eci2ecef.dot(r_sat_eci)
    v_sat_ecef = dcm_eci2ecef.dot(
        v_sat_eci) - np.cross(omega_earth, r_sat_ecef)

    # find relative vector in ecef frame
    rho_ecef = r_sat_ecef - r_site_ecef
    drho_ecef = v_sat_ecef - np.zeros_like(v_sat_ecef)  # site isn't moving
    rho = np.linalg.norm(rho_ecef)

    # convert to SEZ coordinate frame
    dcm_ecef2sez = attitude.rot2(
        np.pi / 2 - lat, 'r').dot(attitude.rot3(lon, 'r'))
    rho_sez = dcm_ecef2sez.dot(rho_ecef)
    drho_sez = dcm_ecef2sez.dot(drho_ecef)

    # find azimuth and eleveation
    temp = np.sqrt(rho_sez[0]**2 + rho_sez[1]**2)

    if temp < small:  # directly over the north pole
        el = np.sign(rho_sez[2]) * halfpi  # \pm 90 deg
    else:
        mag_rho_sez = np.linalg.norm(rho_sez)
        el = np.arcsin(rho_sez[2] / mag_rho_sez)

    if temp < small:
        az = np.arctan2(drho_sez[1], - drho_sez[0])
    else:
        az = np.arctan2(rho_sez[1] / temp, -rho_sez[0] / temp)

    # range, azimuth and elevation rates
    drho = np.dot(rho_sez, drho_sez) / rho

    if np.absolute(temp**2) > small:
        daz = (drho_sez[0] * rho_sez[1] - drho_sez[1] * rho_sez[0]) / temp**2
    else:
        daz = 0

    if np.absolute(temp) > small:
        dele = (drho_sez[2] - drho * np.sin(el)) / temp
    else:
        dele = 0

    az = attitude.normalize(az, 0, constants.twopi)
    return rho, az, el, drho, daz, dele


def rhoazel(sat_eci, site_eci, site_lat, site_lst):
    """
    This function calculates the topcentric range,azimuth and elevation from
    the site vector and satellite position vector.

    Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741

    Inputs:
        sat_eci - satellite ECI position vector (km)
        site_eci - site ECI position vector (km)
        site_lat - site geodetic latitude (radians)
        site_lst - site local sidereal time (radians)

    Outputs:
        rho - range (km)
        az - asimuth (radians)
        el - elevation (radians)

    Globals: None

    Constants: None

    Coupling: 

    References:
        Astro 321 Predict LSN 22 
    """

    site2sat_eci = sat_eci - site_eci

    site2sat_sez = attitude.rot3(-site_lst).dot(site2sat_eci)
    site2sat_sez = attitude.rot2(-(np.pi / 2 - site_lat)).dot(site2sat_sez)

    rho = np.linalg.norm(site2sat_sez)
    el = np.arcsin(site2sat_sez[2] / rho)
    az = attitude.normalize(np.arctan2(site2sat_sez[1], -site2sat_sez[0]), 0,
                            2 * np.pi)[0]

    return rho, az, el


def rhoazel2sez(rho, az, el, drho, daz, dele):
    """This program calculates the range vector in the SEZ system.

    Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741

    Inputs:
        rho - range (km)
        az - azimuth (radians)
        el - elevation (radians)
        drho - range rate (km/sec)
        daz - azimuth rate (radians/sec)
        del - elevation rate (radians/sec)

    Outputs:
        Rho_sez - range vector SEZ (km)
        Drho_sez - range velocity vecotr SEZ (km/sec)

    Globals: None

    Constants: None

    Coupling: None

    References:
        Astro 321 COMFIX
    """

    # Calculate range vector
    rho_s = -rho * np.cos(az) * np.cos(el)
    rho_e = rho * np.sin(az) * np.cos(el)
    rho_z = rho * np.sin(el)

    rho_sez = np.array([rho_s, rho_e, rho_z])

    # Calculate range rate vector
    transform = np.array([[-np.cos(el) * np.cos(az), rho * np.sin(el) * np.cos(az), rho * np.cos(el) * np.sin(az)],
                          [np.cos(el) * np.sin(az), -rho * np.sin(el)
                           * np.sin(az), rho * np.cos(el) * np.cos(az)],
                          [np.sin(el), rho * np.cos(el), 0]])
    drho_sez = transform.dot(np.array([drho, dele, daz]))
    # drho_s=-drho*np.cos(az)*np.cos(el)+rho*daz*np.sin(az)*np.cos(el)+rho*dele*np.cos(az)*np.sin(el)
    # drho_e=drho*np.sin(az)*np.cos(el)+rho*daz*np.cos(az)*np.cos(el)-rho*dele*np.sin(az)*np.sin(el)
    # drho_z=drho*np.sin(el)+rho*dele*np.cos(el)

    # drho_sez=np.array([drho_s,drho_e,drho_z])

    return rho_sez,  drho_sez


def eci2lla(pos_eci, jd):
    """Find LLA for a ECI vector about the Earth
    """
    pass
