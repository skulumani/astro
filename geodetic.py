"""Geodetic transformations
"""
import numpy as np

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
