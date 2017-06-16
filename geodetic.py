"""Geodetic transformations
"""
import numpy as np
from tle_predict.kinematics import attitude

def lla2ecef(lat, lon, alt, r=6378.137, ee=8.1819190842622e-2):
    """Convert latitude, longitude, and altitude to the Earth centered
    Earth fixed frame (ECEF)

    Parameters:
    -----------
    lat : float
        Geodetic latitude (radians)
    lon : float
        Geodetic longitude (radians)
    alt : float
        height about the WGS84 ellipsoid (kilometers)

    Returns
    -------
    ecef : float numpy array (3,)
        ECEF position vector in (kilometers)

    References
    ----------
    """

    N = r / np.sqrt(1 - ee**2 * np.sin(lat)**2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - ee**2) * N + alt) * np.sin(lat)

    return np.array([x, y, z])

def ecef2lla(ecef, r=6378.137, ee=8.1819190842622e-2):
    twopi = 2*np.pi
    tol = 1e-6

    norm_vec = np.linalg.norm(ecef)

    temp = np.sqrt(ecef[0]**2 + ecef[1]**2)
    if temp < tol:
        rtasc = np.sign(ecef[2]) * np.pi * 0.5
    else:
        rtasc = np.arctan2(ecef[1], ecef[0])

    lon = rtasc
    lon = attitude.normalize(lon, 0, 2*np.pi)

    decl = np.arcsin(ecef[2]/ norm_vec)
    latgd = decl

    # iterate to find geodetic latitude
    i = 1
    olddelta = latgd + 10.0
    while np.absolute(olddelta - latgd) >= tol and i < 10:
        oldelta = latgd
        sintemp = np.sin(latgd)
        c = r / np.sqrt(1.0 - ee**2 * sintemp**2)
        latgd = np.arctan2(ecef[2] + c * ee**2 * sintemp, temp)
        i = i +1

    # calculate the height
    if (np.pi/2 - np.absolute(latgd)) > np.pi/180:
        hellp = temp/np.cos(latgd) - c
    else:
        s = c * (1 - ee**2)
        hellp = ecef[2] / np.sin(latgd) - s

    latgc = gd2gc(latgd, ee**2)
    return latgc, latgd, lon, hellp

def gd2gc(latgd, eesqrd=0.081819221456**2):
    """Only valid for locations on the Earth's surface

    Vallado Example 3-1
    """
    latgc = np.arctan( (1 - eesqrd)*np.tan(latgd))
    return latgc

def gc2gd(latgc, eesqrd=0.081819221456**2):
    """Only valid for locations on the Earth's surface

    Vallado Example 3-1
    """
    latgd = np.arctan(np.tan(latgc) / (1 - eesqrd))
    return latgd
