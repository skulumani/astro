"""Geodetic transformations
"""
import numpy as np
from kinematics import attitude
import pdb
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

    Notes
    -----
    We're assuming an oblate spheroid model for the Earth. To transform this
    vector to the inertial ECI frame all you need to do is rotate it by the
    Greenwich Mean Sideral time (GST) or use a fancy ECEF-ECI rotation which
    accounts for the Earth's precession/nutation.

    References
    ----------
    BATE, Roger R, MUELLER, Donald D WHITE, Jerry E. Fundamentals of
    Astrodynamics. Courier Dover Publications, 1971. 
     
    Nice website to verify computations:
    http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    """
    # Normal distance from teh surface to the Z axis along the ellipsoid normal
    N = r / np.sqrt(1 - ee**2 * np.sin(lat)**2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - ee**2) * N + alt) * np.sin(lat)

    return np.array([x, y, z])

def site2eci(lat, alt, lst, r=6378.137, ee=8.1819190842622e-2):
    """Calculate the site vector in the IJK coordinate system.

    Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741

    Inputs:
        sitlat - site latitude (radians)
        lst - local sidereal time (radians)
        sitalt - site altitude (meters)

    Outputs:
        R_site - site vector in IJK system

    Globals: 
        RE, EEsqrd

    Constants: None

    Coupling: None

    References:
        Astro 321 COMFIX
    """
    N = r / np.sqrt(1 - ee**2 * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat)
    z = ((1 - ee**2) * N + alt) * np.sin(lat)
    eci = np.array([x * np.cos(lst), x * np.sin(lst), z])

    return eci

def ecef2lla(ecef, r=6378.137, ee=8.1819190842622e-2):
    """Converts a ECEF vector to the equivalent Lat, longitude and Altitude
    above the reference ellipsoid

    """
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
    site2sat_sez = attitude.rot2(-(np.pi/2 - site_lat)).dot(site2sat_sez)
    
    rho = np.linalg.norm(site2sat_sez)
    el = np.arcsin(site2sat_sez[2] / rho)
    az = attitude.normalize(np.arctan2(site2sat_sez[1], -site2sat_sez[0]), 0, 
            2*np.pi)[0]

    return rho, az, el
